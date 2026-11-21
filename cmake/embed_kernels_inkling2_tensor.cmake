# Concatenates src/kernels/common.h + src/kernels/inkling2_tensor/*.metal into one C++ string literal:
# Inkling-2 Metal 4 tensor_ops kernels compiled once lazily via Gpu::compile_extra("Ink2Tensor", ..., true).
file(READ ${SRC_DIR}/common.h COMMON)
set(ALL "${COMMON}\n")
file(GLOB METALS ${SRC_DIR}/inkling2_tensor/*.metal)
list(SORT METALS)
foreach(m ${METALS})
  file(READ ${m} BODY)
  string(REGEX REPLACE "#include \"[a-z_]+\\.h\"" "" BODY "${BODY}")
  get_filename_component(n ${m} NAME)
  string(APPEND ALL "\n// ---- inkling2_tensor/${n} ----\n${BODY}\n")
endforeach()
file(WRITE ${OUT} "// generated\nnamespace ni { extern const char* const kInk2TensorKernelSource; const char* const kInk2TensorKernelSource = R\"NIKERNEL(${ALL})NIKERNEL\"; }\n")
