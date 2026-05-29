// UNSUPPORTED: true

// TODO(rocmlirTriton): Implement proper error checking for invalid perf_configs and adapt this test.
// TODO(rocmlirTriton): https://amd-hub.atlassian.net/browse/AIROCMLIR-690
// Negative test to deliberately pass incorrect tuning parameters.

// RUN: rocmlir-gen --arch gfx90a -p -mfma=on -t f16 --perf_config "gemm:v1:128,128,4,1,1,4,64,1,2,0,0" | rocmlir-opt -rock-affix-params -verify-diagnostics
// expected-error{{Incorrect KPACK tuning parameter: 2}}

// RUN: rocmlir-gen --arch gfx90a -p -mfma=on -t f16 --perf_config "gemm:v1:128,128,32,1,1,4,64,1,2,0,0" | rocmlir-opt -rock-affix-params -verify-diagnostics
// expected-error{{Incorrect KPACK tuning parameter: 16}}

// RUN: rocmlir-gen --arch gfx90a -p -mfma=on -t f32 --perf_config "gemm:v1:128,128,64,1,1,4,64,1,2,0,0" | rocmlir-opt -rock-affix-params -verify-diagnostics
// expected-error{{Incorrect KPACK tuning parameter: 8}}

// RUN: rocmlir-gen --operation gemm --arch gfx1100 -p -wmma=on -t f16 --perf_config "gemm:v1:256,128,8,1,1,4,128,1,2,0,0" | rocmlir-opt -rock-affix-params -verify-diagnostics
// expected-error{{Wmma instruction selection is not compatible with k.}}
