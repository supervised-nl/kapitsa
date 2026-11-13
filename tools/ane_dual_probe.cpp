// ane_dual_probe: does a second Neural Engine program run alongside the first, or does it wait for it?
//   ane_dual_probe [--c 8192] [--k 5120] [--r 512] [--iters 40] [--max-threads 4]
// Builds one int8 matmul program per thread (y[c, r] = w[c, k] x[k, r], the shape of an FFN projection on a 512-row
// chunk; each program's dequantize scale differs so the daemon compiles and loads each one apart), times N
// synchronous evaluations with 1, 2, ... threads at once, and prints the aggregate TOPS. If two threads together do
// about twice one thread's work, the ANE had idle capacity that one program at a time leaves unused (the M6 reports
// two 16-core devices, H11ANE and H11ANE1, in the I/O registry). Uses no GPU work and loads no model; it takes the GPU
// lock anyway (another run's Neural Engine split would skew it), unless the caller holds it.
#include "ane/ane.h"
#include "bench/bench.h"
#include "util/log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ni;

namespace {

std::string fp16lit(double v) {
  char t[48];
  snprintf(t, sizeof t, "fp16(%a)", v);
  return t;
}
std::string tens(const char* ty, uint64_t r, uint64_t w) {
  return std::string("tensor<") + ty + ", [1, 1, " + std::to_string(r) + ", " + std::to_string(w) + "]>";
}

struct Job {
  ane::Surface x, w, y;
  std::unique_ptr<ane::Program> prog;
  std::shared_ptr<ane::Program::Binding> bind;
};

std::unique_ptr<Job> make_job(uint32_t C, uint32_t K, uint32_t R, int id) {
  auto j = std::make_unique<Job>();
  j->x = ane::make_surface(K, 2048, false);  // 2048 wide, as FfnSplit's lanes (kMaxRows); the program reads R
  j->w = ane::make_surface(C, K, false);
  j->y = ane::make_surface(C, 2048, true);
  std::mt19937 g(1234 + id);
  for (size_t i = 0; i < j->x.bytes; ++i) j->x.ptr()[i] = uint8_t(g() % 31) - 15;
  for (size_t i = 0; i < j->w.bytes; ++i) j->w.ptr()[i] = uint8_t(g() % 31) - 15;
  const std::string scale = fp16lit(1.0 / (128 + id));  // a different program per thread
  std::string b;
  auto line = [&](const std::string& t) { b += "        " + t + ";\n"; };
  line(tens("int8", K, R) + " x_t = tensor_buffer_to_tensor<ios17>(input = x)");
  line(tens("int8", C, K) + " w_t = tensor_buffer_to_tensor<ios17>(input = w)");
  line(tens("fp16", K, R) + " x_d = dequantize(input = x_t, scale = " + scale + ")");
  line(tens("fp16", C, K) + " w_d = dequantize(input = w_t, scale = " + scale + ")");
  line(tens("fp16", C, R) + " m = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w_d, y = x_d)");
  line(j->y.buffer_type(C, R) + " y = tensor_to_tensor_buffer<ios17>(input = m, interleave_factors = " +
       "tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" + j->y.strides(j->y.rows) + "))");
  const std::string mil = "program(1.3)\n{\n    func main<ios18>(" + j->x.buffer_type(K, R) + " x, " +
                          j->w.buffer_type(C, K) + " w) {\n" + b + "    } -> (y);\n}\n";
  const _Float16 one = 1;
  j->prog = std::make_unique<ane::Program>(mil, ane::fp16_blob(&one, 1), "nace-ane-dual-probe");
  const int p = j->prog->proc("main");
  if (p < 0) fatal("ane_dual_probe: no main function");
  std::vector<const ane::Surface*> in;  // in the program's own input order
  for (const std::string& n : j->prog->inputs(p)) in.push_back(n == "w" ? &j->w : &j->x);
  if (id == 0) {
    printf("inputs:");
    for (const std::string& n : j->prog->inputs(p)) printf(" %s", n.c_str());
    printf("\n");
  }
  j->bind = j->prog->bind(p, in, j->y);
  return j;
}

double wall_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t C = 8192, K = 5120, R = 512;
  int iters = 40, max_threads = 4;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const int v = atoi(argv[i + 1]);
    if (k == "--c") C = v;
    else if (k == "--k") K = v;
    else if (k == "--r") R = v;
    else if (k == "--iters") iters = v;
    else if (k == "--max-threads") max_threads = v;
    else fatal("unknown option %s", k.c_str());
  }
  if (!ane::unavailable().empty()) fatal("ane_dual_probe: %s", ane::unavailable().c_str());
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire(nullptr, 3600000)) fatal("ane_dual_probe: could not take the GPU lock");
  std::vector<std::unique_ptr<Job>> jobs;
  for (int t = 0; t < max_threads; ++t) {
    jobs.push_back(make_job(C, K, R, t));
    printf("program %d: %s in %.0f ms, loaded in %.0f ms\n", t, jobs.back()->prog->was_cached ? "cached" : "compiled",
           jobs.back()->prog->compile_ms, jobs.back()->prog->load_ms);
  }
  for (auto& j : jobs)
    for (int w = 0; w < 3; ++w) j->prog->evaluate(*j->bind);
  const double gop = 2.0 * C * K * R / 1e9;
  printf("y[%u, %u] = w[%u, %u] x[%u, %u] int8 -> fp16: %.1f GOP an evaluation, %d evaluations a thread\n", C, R, C,
         K, K, R, gop, iters);
  double one = 0;
  for (int n = 1; n <= max_threads; ++n) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<double> ms(n);
    std::vector<std::thread> th;
    for (int t = 0; t < n; ++t)
      th.emplace_back([&, t] {
        ready++;
        while (!go) std::this_thread::yield();
        const double t0 = wall_ms();
        for (int i = 0; i < iters; ++i) jobs[t]->prog->evaluate(*jobs[t]->bind);
        ms[t] = wall_ms() - t0;
      });
    while (ready < n) std::this_thread::yield();
    const double t0 = wall_ms();
    go = true;
    for (auto& x : th) x.join();
    const double wall = wall_ms() - t0;
    const double tops = gop * iters * n / wall;
    if (n == 1) one = tops;
    printf("%d at once: %.0f ms wall, %.2f ms an evaluation per thread, %.1f TOPS together (%.2fx one)\n", n, wall,
           wall / iters, tops, tops / one);
    fflush(stdout);
  }
  return 0;
}
