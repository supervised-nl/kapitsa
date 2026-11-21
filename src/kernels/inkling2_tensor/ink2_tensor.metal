#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

typedef tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tg_h_t;
typedef tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tg_f_t;

// Inkling-2-Mini Metal 4 (Apple10+ / M5) tensor_ops prefill kernels (all static tgmem <= 16,384 B):
//  - ink2_bf16_mm_t: native BF16/F32 GEMM (64 x 64 x 32 tiles, double-buffered FP32 acc; 16,384 B tgmem)
//  - ink2_msg_gate_prep / ink2_msg_gate_mix: batched cross-pass controller helpers
//  - ink2_moe_bucket / ink2_moe_nvfp4_{w13,w2}_t / ink2_moe_mxfp8_{w13,w2}_t / ink2_moe_reduce8:
//    GPU-bucketed grouped MoE using matmul2d (8,192 B tgmem)
//  - ink2_attn_prefill_t: tiled Flash relative-position sink attention with matmul2d (15,552 B tgmem)

struct Ink2MmTP {
  uint M, N, K, x_ld, y_ld, w_f32, add;
};

struct Ink2MsgPrepP {
  uint T, H;
  float eps;
};

struct Ink2MsgMixP {
  uint n;
};

struct Ink2BucketP {
  uint T, K_top, E;
};

struct Ink2MoeTP {
  uint T, H, I, K_top;
};

struct Ink2AttnPfTP {
  uint T, pos0, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa;
  float scale;
};

NI_INLINE float ink2t_e4m3_16384(uint b) {
  const ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return float(as_type<half>(h)) * 4194304.0f;
}

NI_INLINE half4 ink2t_e2m1_4h(uint bits16, float sc_folded) {
  const ushort4 hv = ushort4(ushort(((bits16 & 0x8u) << 12) | ((bits16 & 0x7u) << 9)),
                             ushort(((bits16 & 0x80u) << 8) | ((bits16 & 0x70u) << 5)),
                             ushort(((bits16 & 0x800u) << 4) | ((bits16 & 0x700u) << 1)),
                             ushort((bits16 & 0x8000u) | ((bits16 & 0x7000u) >> 3)));
  return half4(float4(as_type<half4>(hv)) * sc_folded);
}

NI_INLINE float ink2t_e4m3(uint b) {
  const ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return float(as_type<half>(h)) * 256.0f;
}

NI_INLINE float ink2t_ue8m0(uint b) {
  return b == 0u ? 5.8774718e-39f : as_type<float>(b << 23);
}

NI_INLINE float ink2t_situ(float g, float u) {
  return (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
         (25.0f * metal::precise::tanh(u * 0.04f));
}

// ---- 1. Tiled Metal 4 BF16/F32 GEMM (FP16 tensor tiles -> FP32 accumulate): Y[M, N] (+)= X[M, K] * W[N, K]^T ----
[[kernel, max_total_threads_per_threadgroup(128)]] void ink2_bf16_mm_t(
    const device float* X [[buffer(0)]], const device void* W [[buffer(1)]], device float* Y [[buffer(2)]],
    constant Ink2MmTP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint BM = 64, BN = 64, BK = 32, SM = 32, SN = 32;
  const uint m0 = tg.y * BM, n0 = tg.x * BN;
  if (m0 >= p.M || n0 >= p.N) return;
  const uint sm = sg >> 1, sn = sg & 1;

  // 16,384 B shared scratch: double-buffered xs (8,192 B) + ws (8,192 B).
  threadgroup half xs[2 * BM * BK];
  threadgroup half ws[2 * BN * BK];
  static_assert(sizeof(xs) + sizeof(ws) <= 16384, "ink2_bf16_mm_t tgmem exceeds 16 KB");

  constexpr auto desc =
      matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  auto tX0 = tg_h_t(xs + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tX1 = tg_h_t(xs + BM * BK + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tW0 = tg_h_t(ws + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto tW1 = tg_h_t(ws + BN * BK + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto acc = op.template get_destination_cooperative_tensor<decltype(tX0), decltype(tW0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  // 128 threads load 64x32 X (16 elements per thread) and 64x32 W (16 elements per thread)
  const uint lr = tid >> 1, lc = (tid & 1u) * 16;
  const uint xm = m0 + lr;
  const bool x_ok = xm < p.M;
  const uint wn = n0 + lr;
  const bool w_ok = wn < p.N;

  auto load_stage = [&](uint b, uint k0) {
    threadgroup half* xdst = xs + b * BM * BK + lr * BK + lc;
    if (x_ok && k0 + lc + 15 < p.K) {
      const device float4* xp = (const device float4*)(X + size_t(xm) * p.x_ld + k0 + lc);
      *(threadgroup half4*)(xdst + 0) = half4(xp[0]);
      *(threadgroup half4*)(xdst + 4) = half4(xp[1]);
      *(threadgroup half4*)(xdst + 8) = half4(xp[2]);
      *(threadgroup half4*)(xdst + 12) = half4(xp[3]);
    } else {
      for (uint q = 0; q < 16; ++q)
        xdst[q] = (x_ok && k0 + lc + q < p.K) ? half(X[size_t(xm) * p.x_ld + k0 + lc + q]) : 0.0h;
    }

    threadgroup half* wdst = ws + b * BN * BK + lr * BK + lc;
    if (w_ok && k0 + lc + 15 < p.K) {
      if (p.w_f32) {
        const device float4* wp = (const device float4*)((const device float*)W + size_t(wn) * p.K + k0 + lc);
        for (uint g = 0; g < 4; ++g) *(threadgroup half4*)(wdst + g * 4) = half4(wp[g]);
      } else {
        const device uint4* wp = (const device uint4*)((const device bfloat*)W + size_t(wn) * p.K + k0 + lc);
        const uint4 a = wp[0], c = wp[1];
        *(threadgroup half4*)(wdst + 0) = half4(
            half(as_type<float>(a.x << 16)), half(as_type<float>(a.x & 0xFFFF0000u)),
            half(as_type<float>(a.y << 16)), half(as_type<float>(a.y & 0xFFFF0000u)));
        *(threadgroup half4*)(wdst + 4) = half4(
            half(as_type<float>(a.z << 16)), half(as_type<float>(a.z & 0xFFFF0000u)),
            half(as_type<float>(a.w << 16)), half(as_type<float>(a.w & 0xFFFF0000u)));
        *(threadgroup half4*)(wdst + 8) = half4(
            half(as_type<float>(c.x << 16)), half(as_type<float>(c.x & 0xFFFF0000u)),
            half(as_type<float>(c.y << 16)), half(as_type<float>(c.y & 0xFFFF0000u)));
        *(threadgroup half4*)(wdst + 12) = half4(
            half(as_type<float>(c.z << 16)), half(as_type<float>(c.z & 0xFFFF0000u)),
            half(as_type<float>(c.w << 16)), half(as_type<float>(c.w & 0xFFFF0000u)));
      }
    } else {
      for (uint q = 0; q < 16; ++q) {
        float v = 0.0f;
        if (w_ok && k0 + lc + q < p.K) {
          const size_t idx = size_t(wn) * p.K + k0 + lc + q;
          v = p.w_f32 ? ((const device float*)W)[idx] : float(((const device bfloat*)W)[idx]);
        }
        wdst[q] = half(v);
      }
    }
  };

  load_stage(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint steps = (p.K + BK - 1) / BK;
  for (uint st = 0; st < steps; ++st) {
    if (st + 1u < steps) load_stage((st + 1u) & 1u, (st + 1u) * BK);
    if (st & 1u) op.run(tX1, tW1, acc);
    else op.run(tX0, tW0, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (m0 + sm * SM + SM <= p.M && n0 + sn * SN + SN <= p.N && (p.N & 3u) == 0u && (p.y_ld & 3u) == 0u) {
    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        Y + size_t(m0 + sm * SM) * p.y_ld + n0 + sn * SN, dextents<int32_t, 2>(SN, SM),
        array<int32_t, 2>{1, int32_t(p.y_ld)});
    if (p.add) {
      auto c = op.template get_destination_cooperative_tensor<decltype(tX0), decltype(tW0), float>();
      c.load(tY);
      _Pragma("clang loop unroll(full)")
      for (uint16_t i = 0; i < c.get_capacity(); ++i) c[i] = c[i] + acc[i];
      c.store(tY);
    } else {
      acc.store(tY);
    }
    return;
  }
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto ix = acc.get_multidimensional_index(i);
    const uint n = n0 + sn * SN + ix[0], m = m0 + sm * SM + ix[1];
    if (m >= p.M || n >= p.N) continue;
    const size_t o = size_t(m) * p.y_ld + n;
    const float v = acc[i];
    Y[o] = p.add ? (Y[o] + v) : v;
  }
}

// ---- 2. Batched Cross-Pass MessageGate Helpers (T >= 32) ----
[[kernel]] void ink2_msg_gate_prep(const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
                                     device float* cat [[buffer(2)]], constant Ink2MsgPrepP& p [[buffer(3)]],
                                     uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* hr = h + size_t(t) * p.H;
  const device float* mr = m + size_t(t) * p.H;
  float hs[5], ms[5];
  float ssh = 0.0f, ssm = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float hv = hr[c], mv = mr[c];
    hs[k] = hv;
    ms[k] = mv;
    ssh += hv * hv;
    ssm += mv * mv;
  }
  ssh = tg_sum(ssh, scratch, tid, ntg);
  ssm = tg_sum(ssm, scratch, tid, ntg);
  const float invh = metal::precise::rsqrt(ssh / float(p.H) + p.eps);
  const float invm = metal::precise::rsqrt(ssm / float(p.H) + p.eps);
  device float* d0 = cat + size_t(t) * (2 * p.H);
  device float* d1 = d0 + p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    d0[c] = hs[k] * invh;
    d1[c] = ms[k] * invm;
  }
}

[[kernel]] void ink2_msg_gate_mix(const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
                                    const device float* f_in [[buffer(2)]], const device float* g_in [[buffer(3)]],
                                    device float* out [[buffer(4)]], constant Ink2MsgMixP& p [[buffer(5)]],
                                    uint i [[thread_position_in_grid]]) {
  const uint idx = i * 4;
  if (idx + 3 < p.n) {
    const float4 hv = *(const device float4*)(h + idx);
    const float4 mv = *(const device float4*)(m + idx);
    const float4 fv = *(const device float4*)(f_in + idx);
    const float4 gv = *(const device float4*)(g_in + idx);
    const float4 fg = float4(metal::precise::tanh(fv.x), metal::precise::tanh(fv.y), metal::precise::tanh(fv.z),
                             metal::precise::tanh(fv.w));
    const float4 gn = float4(metal::precise::tanh(gv.x), metal::precise::tanh(gv.y), metal::precise::tanh(gv.z),
                             metal::precise::tanh(gv.w));
    *(device float4*)(out + idx) = (1.0f - fg) * hv + gn * mv;
  } else {
    for (uint q = idx; q < p.n; ++q)
      out[q] = (1.0f - metal::precise::tanh(f_in[q])) * h[q] + metal::precise::tanh(g_in[q]) * m[q];
  }
}

// ---- 3. GPU-Bucketed Grouped MoE on Metal 4 (T >= 32) ----
[[kernel, max_total_threads_per_threadgroup(128)]] void ink2_moe_bucket(
    const device uint* eid [[buffer(0)]], const device float* ew [[buffer(1)]], device uint* tiles [[buffer(2)]],
    device uint* ntiles [[buffer(3)]], device uint* st_tok [[buffer(4)]], device uint* st_pair [[buffer(5)]],
    device float* st_ew [[buffer(6)]], constant Ink2BucketP& p [[buffer(7)]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint BM = 32;
  const uint R = p.T * p.K_top;
  threadgroup atomic_uint counts[128];
  threadgroup atomic_uint cursors[128];
  threadgroup uint exp_off[128];
  threadgroup uint tcnt[128];
  threadgroup uint toff[128];

  atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < R; i += 128) atomic_fetch_add_explicit(&counts[eid[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint my_cnt = atomic_load_explicit(&counts[tid], memory_order_relaxed);
  const uint my_tcnt = (my_cnt + BM - 1) / BM;
  exp_off[tid] = my_cnt;
  tcnt[tid] = my_tcnt;
  toff[tid] = my_tcnt;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Inclusive Kogge-Stone scan over 128 experts in threadgroup memory
  for (uint stride = 1; stride < 128; stride <<= 1) {
    const uint e_add = tid >= stride ? exp_off[tid - stride] : 0u;
    const uint t_add = tid >= stride ? toff[tid - stride] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    exp_off[tid] += e_add;
    toff[tid] += t_add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint my_exp_start = exp_off[tid] - my_cnt;
  const uint my_t_start = toff[tid] - my_tcnt;
  atomic_store_explicit(&cursors[tid], my_exp_start, memory_order_relaxed);
  if (tid == 127) ntiles[0] = toff[127];

  for (uint ti = 0; ti < my_tcnt; ++ti) {
    const uint off = ti * BM;
    const uint t_idx = my_t_start + ti;
    tiles[t_idx * 3 + 0] = tid;
    tiles[t_idx * 3 + 1] = my_exp_start + off;
    tiles[t_idx * 3 + 2] = min(BM, my_cnt - off);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < R; i += 128) {
    const uint e = eid[i];
    const uint dst = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
    st_tok[dst] = i / p.K_top;
    st_pair[dst] = i;
    st_ew[dst] = ew[i];
  }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void ink2_moe_nvfp4_w13_t(
    const device float* x [[buffer(0)]], const device uchar* w13 [[buffer(1)]], const device uchar* sc [[buffer(2)]],
    const device float* outer [[buffer(3)]], const device uint* tiles [[buffer(4)]],
    const device uint* ntiles [[buffer(5)]], const device uint* st_tok [[buffer(6)]], device float* act [[buffer(7)]],
    constant Ink2MoeTP& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  if (tg.y >= ntiles[0]) return;
  constexpr uint BN_P = 32, BK = 32, SM = 16;
  const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2];
  if (rows == 0) return;
  const uint j0 = tg.x * BN_P;
  if (j0 >= p.I) return;
  const uint sm = sg >> 1, sn = sg & 1;

  // 8,192 B shared scratch: xs (2,048 B) + wsg (2,048 B) + wsu (2,048 B) in K-loop, then sg_buf + su_buf (8,192 B).
  threadgroup float4 tg_stg[512];
  static_assert(sizeof(tg_stg) <= 16384, "ink2_moe_nvfp4_w13_t tgmem exceeds 16 KB");
  threadgroup half* xs = (threadgroup half*)tg_stg;
  threadgroup half* wsg = (threadgroup half*)(tg_stg + 128);
  threadgroup half* wsu = (threadgroup half*)(tg_stg + 256);
  threadgroup float* sg_buf = (threadgroup float*)tg_stg;
  threadgroup float* su_buf = (threadgroup float*)(tg_stg + 256);

  constexpr auto desc =
      matmul2d_descriptor(SM, BN_P, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  auto tX = tg_h_t(xs + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tW = tg_h_t(sn ? wsu : wsg, dextents<int32_t, 2>(BK, BN_P));
  auto acc = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  const uint xr = tid >> 2, xc = (tid & 3u) * 8;
  const uint tok = xr < rows ? st_tok[m0 + xr] : 0xffffffffu;
  const uint wr = tid >> 1, wk = (tid & 1u) * 16;
  const uint half_h = p.H >> 1, nblk = p.H >> 4;
  const size_t wrow = size_t(e) * (2 * p.I) + 2 * j0 + wr;
  const bool w_ok = j0 + (wr >> 1) < p.I;
  threadgroup half* wdst = (wr & 1u ? wsu : wsg) + (wr >> 1) * BK + wk;

  for (uint k0 = 0; k0 < p.H; k0 += BK) {
    threadgroup half* xdst = xs + xr * BK + xc;
    if (tok != 0xffffffffu) {
      const float4 v0 = *(const device float4*)(x + size_t(tok) * p.H + k0 + xc);
      const float4 v1 = *(const device float4*)(x + size_t(tok) * p.H + k0 + xc + 4);
      *(threadgroup half4*)(xdst + 0) = half4(v0);
      *(threadgroup half4*)(xdst + 4) = half4(v1);
    } else {
      *(threadgroup uint4*)(xdst) = uint4(0);
    }

    if (w_ok) {
      const uint blk = (k0 + wk) >> 4;
      const float sc_f = ink2t_e4m3_16384(sc[wrow * nblk + blk]);
      const uint2 pw = *(const device uint2*)(w13 + wrow * half_h + blk * 8);
      *(threadgroup half4*)(wdst + 0) = ink2t_e2m1_4h(pw.x, sc_f);
      *(threadgroup half4*)(wdst + 4) = ink2t_e2m1_4h(pw.x >> 16, sc_f);
      *(threadgroup half4*)(wdst + 8) = ink2t_e2m1_4h(pw.y, sc_f);
      *(threadgroup half4*)(wdst + 12) = ink2t_e2m1_4h(pw.y >> 16, sc_f);
    } else {
      *(threadgroup uint4*)(wdst + 0) = uint4(0);
      *(threadgroup uint4*)(wdst + 8) = uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sm * SM < rows) op.run(tX, tW, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  auto tOut = tg_f_t((sn ? su_buf : sg_buf) + sm * SM * BN_P, dextents<int32_t, 2>(BN_P, SM));
  acc.store(tOut);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float os = outer[e];
  for (uint i = tid; i < rows * BN_P; i += 128) {
    const uint r = i / BN_P, c = i % BN_P;
    if (j0 + c < p.I) act[size_t(m0 + r) * p.I + j0 + c] = ink2t_situ(sg_buf[i] * os, su_buf[i] * os);
  }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void ink2_moe_nvfp4_w2_t(
    const device float* act [[buffer(0)]], const device uchar* w2 [[buffer(1)]], const device uchar* sc [[buffer(2)]],
    const device float* outer [[buffer(3)]], const device uint* tiles [[buffer(4)]],
    const device uint* ntiles [[buffer(5)]], const device uint* st_pair [[buffer(6)]],
    const device float* st_ew [[buffer(7)]], device float* pair_out [[buffer(8)]], constant Ink2MoeTP& p [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (tg.y >= ntiles[0]) return;
  constexpr uint BN = 64, BK = 32, SM = 16, SN = 32;
  const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2];
  if (rows == 0) return;
  const uint n0 = tg.x * BN;
  if (n0 >= p.H) return;
  const uint sm = sg >> 1, sn = sg & 1;

  // 8,192 B shared scratch: xs (2,048 B) + ws (4,096 B) in K-loop, then out_stg (8,192 B).
  threadgroup float4 tg_stg[512];
  static_assert(sizeof(tg_stg) <= 16384, "ink2_moe_nvfp4_w2_t tgmem exceeds 16 KB");
  threadgroup half* xs = (threadgroup half*)tg_stg;
  threadgroup half* ws = (threadgroup half*)(tg_stg + 128);
  threadgroup float* out_stg = (threadgroup float*)tg_stg;

  constexpr auto desc =
      matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  auto tX = tg_h_t(xs + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tW = tg_h_t(ws + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto acc = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  const uint xr = tid >> 2, xc = (tid & 3u) * 8;
  const uint wr = tid >> 1, wk = (tid & 1u) * 16;
  const uint half_i = p.I >> 1, nblk = p.I >> 4;
  const uint wn = n0 + wr;
  const bool w_ok = wn < p.H;
  const size_t wrow = size_t(e) * p.H + wn;
  threadgroup half* wdst = ws + wr * BK + wk;

  for (uint k0 = 0; k0 < p.I; k0 += BK) {
    threadgroup half* xdst = xs + xr * BK + xc;
    if (xr < rows) {
      const float4 v0 = *(const device float4*)(act + size_t(m0 + xr) * p.I + k0 + xc);
      const float4 v1 = *(const device float4*)(act + size_t(m0 + xr) * p.I + k0 + xc + 4);
      *(threadgroup half4*)(xdst + 0) = half4(v0);
      *(threadgroup half4*)(xdst + 4) = half4(v1);
    } else {
      *(threadgroup uint4*)(xdst) = uint4(0);
    }

    if (w_ok) {
      const uint blk = (k0 + wk) >> 4;
      const float sc_f = ink2t_e4m3_16384(sc[wrow * nblk + blk]);
      const uint2 pw = *(const device uint2*)(w2 + wrow * half_i + blk * 8);
      *(threadgroup half4*)(wdst + 0) = ink2t_e2m1_4h(pw.x, sc_f);
      *(threadgroup half4*)(wdst + 4) = ink2t_e2m1_4h(pw.x >> 16, sc_f);
      *(threadgroup half4*)(wdst + 8) = ink2t_e2m1_4h(pw.y, sc_f);
      *(threadgroup half4*)(wdst + 12) = ink2t_e2m1_4h(pw.y >> 16, sc_f);
    } else {
      *(threadgroup uint4*)(wdst + 0) = uint4(0);
      *(threadgroup uint4*)(wdst + 8) = uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sm * SM < rows) op.run(tX, tW, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  auto tOut = tg_f_t(out_stg + sm * SM * BN + sn * SN, dextents<int32_t, 2>(SN, SM), array<int32_t, 2>{1, BN});
  acc.store(tOut);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float os = outer[e];
  for (uint e4 = tid; e4 < (rows * BN) / 4; e4 += 128) {
    const uint r = e4 >> 4, c4 = (e4 & 15u) * 4;
    if (n0 + c4 < p.H) {
      const uint pair = st_pair[m0 + r];
      const float scale = os * st_ew[m0 + r];
      *(device float4*)(pair_out + size_t(pair) * p.H + n0 + c4) =
          *(const threadgroup float4*)(out_stg + r * BN + c4) * scale;
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void ink2_moe_mxfp8_w13_t(
    const device float* x [[buffer(0)]], const device uchar* w13 [[buffer(1)]], const device uchar* sc [[buffer(2)]],
    const device uint* tiles [[buffer(3)]], const device uint* ntiles [[buffer(4)]],
    const device uint* st_tok [[buffer(5)]], device float* act [[buffer(6)]], constant Ink2MoeTP& p [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (tg.y >= ntiles[0]) return;
  constexpr uint BN_P = 32, BK = 32, SM = 16;
  const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2];
  if (rows == 0) return;
  const uint j0 = tg.x * BN_P;
  if (j0 >= p.I) return;
  const uint sm = sg >> 1, sn = sg & 1;

  threadgroup float4 tg_stg[512];
  static_assert(sizeof(tg_stg) <= 16384, "ink2_moe_mxfp8_w13_t tgmem exceeds 16 KB");
  threadgroup half* xs = (threadgroup half*)tg_stg;
  threadgroup half* wsg = (threadgroup half*)(tg_stg + 128);
  threadgroup half* wsu = (threadgroup half*)(tg_stg + 256);
  threadgroup float* sg_buf = (threadgroup float*)tg_stg;
  threadgroup float* su_buf = (threadgroup float*)(tg_stg + 256);

  constexpr auto desc =
      matmul2d_descriptor(SM, BN_P, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  auto tX = tg_h_t(xs + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tW = tg_h_t(sn ? wsu : wsg, dextents<int32_t, 2>(BK, BN_P));
  auto acc = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  const uint xr = tid >> 2, xc = (tid & 3u) * 8;
  const uint tok = xr < rows ? st_tok[m0 + xr] : 0xffffffffu;
  const uint wr = tid >> 1, wk = (tid & 1u) * 16;
  const uint nblk = p.H >> 5;
  const size_t wrow = size_t(e) * (2 * p.I) + 2 * j0 + wr;
  const bool w_ok = j0 + (wr >> 1) < p.I;
  threadgroup half* wdst = (wr & 1u ? wsu : wsg) + (wr >> 1) * BK + wk;

  for (uint k0 = 0; k0 < p.H; k0 += BK) {
    threadgroup half* xdst = xs + xr * BK + xc;
    if (tok != 0xffffffffu) {
      const float4 v0 = *(const device float4*)(x + size_t(tok) * p.H + k0 + xc);
      const float4 v1 = *(const device float4*)(x + size_t(tok) * p.H + k0 + xc + 4);
      *(threadgroup half4*)(xdst + 0) = half4(v0);
      *(threadgroup half4*)(xdst + 4) = half4(v1);
    } else {
      *(threadgroup uint4*)(xdst) = uint4(0);
    }

    if (w_ok) {
      const float sc_b = ink2t_ue8m0(sc[wrow * nblk + (k0 >> 5)]);
      const uint4 pw = *(const device uint4*)(w13 + wrow * p.H + k0 + wk);
      for (uint g = 0; g < 4; ++g) {
        const uint w = pw[g];
        wdst[g * 4 + 0] = half(ink2t_e4m3(w & 0xFFu) * sc_b);
        wdst[g * 4 + 1] = half(ink2t_e4m3((w >> 8) & 0xFFu) * sc_b);
        wdst[g * 4 + 2] = half(ink2t_e4m3((w >> 16) & 0xFFu) * sc_b);
        wdst[g * 4 + 3] = half(ink2t_e4m3(w >> 24) * sc_b);
      }
    } else {
      *(threadgroup uint4*)(wdst + 0) = uint4(0);
      *(threadgroup uint4*)(wdst + 8) = uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sm * SM < rows) op.run(tX, tW, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  auto tOut = tg_f_t((sn ? su_buf : sg_buf) + sm * SM * BN_P, dextents<int32_t, 2>(BN_P, SM));
  acc.store(tOut);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < rows * BN_P; i += 128) {
    const uint r = i / BN_P, c = i % BN_P;
    if (j0 + c < p.I) act[size_t(m0 + r) * p.I + j0 + c] = ink2t_situ(sg_buf[i], su_buf[i]);
  }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void ink2_moe_mxfp8_w2_t(
    const device float* act [[buffer(0)]], const device uchar* w2 [[buffer(1)]], const device uchar* sc [[buffer(2)]],
    const device uint* tiles [[buffer(3)]], const device uint* ntiles [[buffer(4)]],
    const device uint* st_pair [[buffer(5)]], const device float* st_ew [[buffer(6)]],
    device float* pair_out [[buffer(7)]], constant Ink2MoeTP& p [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (tg.y >= ntiles[0]) return;
  constexpr uint BN = 64, BK = 32, SM = 16, SN = 32;
  const uint e = tiles[tg.y * 3 + 0], m0 = tiles[tg.y * 3 + 1], rows = tiles[tg.y * 3 + 2];
  if (rows == 0) return;
  const uint n0 = tg.x * BN;
  if (n0 >= p.H) return;
  const uint sm = sg >> 1, sn = sg & 1;

  threadgroup float4 tg_stg[512];
  static_assert(sizeof(tg_stg) <= 16384, "ink2_moe_mxfp8_w2_t tgmem exceeds 16 KB");
  threadgroup half* xs = (threadgroup half*)tg_stg;
  threadgroup half* ws = (threadgroup half*)(tg_stg + 128);
  threadgroup float* out_stg = (threadgroup float*)tg_stg;

  constexpr auto desc =
      matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;

  auto tX = tg_h_t(xs + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tW = tg_h_t(ws + sn * SN * BK, dextents<int32_t, 2>(BK, SN));
  auto acc = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  const uint xr = tid >> 2, xc = (tid & 3u) * 8;
  const uint wr = tid >> 1, wk = (tid & 1u) * 16;
  const uint nblk = p.I >> 5;
  const uint wn = n0 + wr;
  const bool w_ok = wn < p.H;
  const size_t wrow = size_t(e) * p.H + wn;
  threadgroup half* wdst = ws + wr * BK + wk;

  for (uint k0 = 0; k0 < p.I; k0 += BK) {
    threadgroup half* xdst = xs + xr * BK + xc;
    if (xr < rows) {
      const float4 v0 = *(const device float4*)(act + size_t(m0 + xr) * p.I + k0 + xc);
      const float4 v1 = *(const device float4*)(act + size_t(m0 + xr) * p.I + k0 + xc + 4);
      *(threadgroup half4*)(xdst + 0) = half4(v0);
      *(threadgroup half4*)(xdst + 4) = half4(v1);
    } else {
      *(threadgroup uint4*)(xdst) = uint4(0);
    }

    if (w_ok) {
      const float sc_b = ink2t_ue8m0(sc[wrow * nblk + (k0 >> 5)]);
      const uint4 pw = *(const device uint4*)(w2 + wrow * p.I + k0 + wk);
      for (uint g = 0; g < 4; ++g) {
        const uint w = pw[g];
        wdst[g * 4 + 0] = half(ink2t_e4m3(w & 0xFFu) * sc_b);
        wdst[g * 4 + 1] = half(ink2t_e4m3((w >> 8) & 0xFFu) * sc_b);
        wdst[g * 4 + 2] = half(ink2t_e4m3((w >> 16) & 0xFFu) * sc_b);
        wdst[g * 4 + 3] = half(ink2t_e4m3(w >> 24) * sc_b);
      }
    } else {
      *(threadgroup uint4*)(wdst + 0) = uint4(0);
      *(threadgroup uint4*)(wdst + 8) = uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sm * SM < rows) op.run(tX, tW, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  auto tOut = tg_f_t(out_stg + sm * SM * BN + sn * SN, dextents<int32_t, 2>(SN, SM), array<int32_t, 2>{1, BN});
  acc.store(tOut);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint e4 = tid; e4 < (rows * BN) / 4; e4 += 128) {
    const uint r = e4 >> 4, c4 = (e4 & 15u) * 4;
    if (n0 + c4 < p.H) {
      const uint pair = st_pair[m0 + r];
      const float wt = st_ew[m0 + r];
      *(device float4*)(pair_out + size_t(pair) * p.H + n0 + c4) =
          *(const threadgroup float4*)(out_stg + r * BN + c4) * wt;
    }
  }
}

[[kernel]] void ink2_moe_reduce8(const device float* pair_out [[buffer(0)]], device float* out [[buffer(1)]],
                                   constant Ink2MoeTP& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x * 4, t = gid.y;
  if (c >= p.H || t >= p.T) return;
  const size_t base = size_t(t) * p.K_top * p.H + c;
  float4 sum = float4(0.0f);
  for (uint slot = 0; slot < p.K_top; ++slot) sum += *(const device float4*)(pair_out + base + size_t(slot) * p.H);
  *(device float4*)(out + size_t(t) * p.H + c) = sum;
}

// ---- 4. Metal 4 Tiled Flash Relative-Position Sink Attention Prefill (T >= 16, <= 16 KB tgmem) ----
[[kernel, max_total_threads_per_threadgroup(128)]] void ink2_attn_prefill_t(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    const device float* rb [[buffer(3)]], const device float* sinks [[buffer(4)]], const device float* gate [[buffer(5)]],
    device float* out [[buffer(6)]], constant Ink2AttnPfTP& p [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BM = 16, BK = 32, D = 128;
  const uint row0 = tg.x * BM, h = tg.y;
  if (row0 >= p.T) return;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint valid_q = min(BM, p.T - row0);
  const uint k_first = p.is_swa ? uint(max(0, int(p.pos0 + row0) + 1 - int(p.swa))) : 0u;
  const uint k_last = p.pos0 + row0 + valid_q - 1;

  // Total static tgmem: 8,192 + 4,096 + 2,048 + 1,024 + 192 = 15,552 B (<= 16,384 B; passes MTL_DEBUG_LAYER=1).
  threadgroup float4 kv_or_o[512];     // 8,192 B: K/V [32, 128] half during loop, then O [16, 128] float at epilogue
  threadgroup half q_tile[BM * D];     // 4,096 B
  threadgroup float s_tile[BM * BK];   // 2,048 B
  threadgroup half probs[BM * BK];     // 1,024 B
  threadgroup float sprior[BM];        // 64 B
  threadgroup float tg_m[BM], tg_l[BM];// 128 B
  static_assert(sizeof(kv_or_o) + sizeof(q_tile) + sizeof(s_tile) + sizeof(probs) + sizeof(sprior) + sizeof(tg_m) +
                        sizeof(tg_l) <=
                    16384,
                "ink2_attn_prefill_t tgmem exceeds 16 KB");
  threadgroup half* kv_tile = (threadgroup half*)kv_or_o;
  threadgroup float* o_tile = (threadgroup float*)kv_or_o;

  // Stage Q[row0..row0+valid_q-1, h, :] * scale into q_tile[16 * 128] (half)
  for (uint i = tid; i < BM * (D / 4); i += 128) {
    const uint r = i >> 5, c4 = (i & 31u) * 4;
    half4 hv = half4(0.0h);
    if (r < valid_q) {
      const float4 qv = *(const device float4*)(q + (size_t(row0 + r) * p.Hq + h) * D + c4) * p.scale;
      hv = half4(qv);
    }
    *(threadgroup half4*)(q_tile + r * D + c4) = hv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto qk_desc = matmul2d_descriptor(16, 16, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_desc =
      matmul2d_descriptor(16, 32, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroup> qk_op;
  matmul2d<pv_desc, execution_simdgroup> pv_op;

  auto tQ = tg_h_t(q_tile, dextents<int32_t, 2>(D, BM));
  auto tK = tg_h_t(kv_tile + (sg & 1u) * 16 * D, dextents<int32_t, 2>(D, 16));
  auto tS = tg_f_t(s_tile + (sg & 1u) * 16, dextents<int32_t, 2>(16, BM), array<int32_t, 2>{1, int32_t(BK)});
  auto tP = tg_h_t(probs, dextents<int32_t, 2>(BK, BM));
  auto tV = tg_h_t(kv_tile + sg * 32, dextents<int32_t, 2>(32, BK), array<int32_t, 2>{1, int32_t(D)});
  auto acc_o = pv_op.template get_destination_cooperative_tensor<decltype(tP), decltype(tV), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc_o.get_capacity(); ++i) acc_o[i] = 0.0f;

  // Simdgroup sg maintains online softmax statistics for rows sg, sg+4, sg+8, sg+12
  float row_m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  float row_l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const uint col = sg * 32 + lane;

  for (uint k0 = k_first; k0 <= k_last; k0 += BK) {
    // 1. Load K[k0 .. k0+31]
    for (uint i = tid; i < BK * (D / 8); i += 128) {
      const uint r = i >> 4, c8 = (i & 15u) * 8, kp = k0 + r;
      uint4 kv = uint4(0);
      if (kp <= k_last) {
        const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
        kv = *(const device uint4*)(kc + (size_t(slot) * p.Hkv + hkv) * D + c8);
      }
      *(threadgroup uint4*)(kv_tile + r * D + c8) = kv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Q @ K^T -> s_tile[16, 32] (2 simdgroups cover the 32 key columns in 2 x 16)
    if (sg < 2u) qk_op.run(tQ, tK, tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Relative bias + causal/SWA mask + online softmax -> probs[16, 32] & sprior[16]
    for (uint k = 0; k < 4; ++k) {
      const uint r = sg + k * 4;
      const uint qrow = row0 + r;
      const int qpos = int(p.pos0 + qrow);
      const device float* rbh = rb + (size_t(min(qrow, p.T - 1)) * p.Hq + h) * p.rel_extent;
      float sc0 = -INFINITY;
      if (r < valid_q) {
        const int kp0 = int(k0 + lane);
        const int d0 = qpos - kp0;
        if (kp0 <= int(k_last) && d0 >= 0 && (!p.is_swa || d0 < int(p.swa))) {
          sc0 = s_tile[r * BK + lane];
          if (d0 < int(p.rel_extent)) sc0 += rbh[d0];
        }
      }
      const float rmax = simd_max(sc0);
      const float prev_m = row_m[k];
      const float nm = max(prev_m, rmax);
      const float prior = (isfinite(prev_m) && isfinite(nm)) ? metal::precise::exp(prev_m - nm) : 0.0f;
      const float w0 = isfinite(sc0) ? metal::precise::exp(sc0 - nm) : 0.0f;
      probs[r * BK + lane] = half(w0);
      row_l[k] = fma(row_l[k], prior, simd_sum(w0));
      row_m[k] = nm;
      if (lane == 0) sprior[r] = prior;
    }

    // 4. Load V[k0 .. k0+31] into kv_tile (while ensuring sprior writes are visible before rescaling acc_o)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * (D / 8); i += 128) {
      const uint r = i >> 4, c8 = (i & 15u) * 8, kp = k0 + r;
      uint4 vv = uint4(0);
      if (kp <= k_last) {
        const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
        vv = *(const device uint4*)(vc + (size_t(slot) * p.Hkv + hkv) * D + c8);
      }
      *(threadgroup uint4*)(kv_tile + r * D + c8) = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Rescale running accumulator by sprior[row], then acc_o += probs @ V
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < acc_o.get_capacity(); ++i) {
      if (!acc_o.is_valid_element(i)) continue;
      const auto ix = acc_o.get_multidimensional_index(i);
      acc_o[i] *= sprior[ix[1]];
    }
    pv_op.run(tP, tV, acc_o);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store final O[16, 128] into kv_or_o (dead kv_tile space)
  auto tO = tg_f_t(o_tile + sg * 32, dextents<int32_t, 2>(32, BM), array<int32_t, 2>{1, int32_t(D)});
  acc_o.store(tO);
  if (lane == 0) {
    for (uint k = 0; k < 4; ++k) {
      const uint r = sg + k * 4;
      tg_m[r] = row_m[k];
      tg_l[r] = row_l[k];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint r = 0; r < valid_q; ++r) {
    const size_t qoff = (size_t(row0 + r) * p.Hq + h) * D + col;
    const float sk = sinks[size_t(row0 + r) * p.Hq + h];
    const float mf = tg_m[r], lf = tg_l[r];
    const float nm = max(mf, sk);
    const float corr = isfinite(mf) ? metal::precise::exp(mf - nm) : 0.0f;
    const float denom = lf * corr + metal::precise::exp(sk - nm);
    const float g = gate[qoff];
    out[qoff] = (o_tile[r * D + col] * corr / denom) * sigmoid(g);
  }
}
