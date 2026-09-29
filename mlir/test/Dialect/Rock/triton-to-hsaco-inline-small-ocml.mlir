// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1200' %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

// The call-count threshold alone is insufficient to trigger outlining. This
// synthetic OCML-prefixed internal function has a two-instruction LLVM body,
// so 128 call sites stay below the duplicated-instruction budget and inline.
//
// CHECK: // -----// LLVM IR Dump //----- //
// CHECK-NOT: @__ocml_small_f32
// CHECK-LABEL: define amdgpu_kernel void @kernel
// CHECK-COUNT-128: fadd float
// CHECK-NOT: @__ocml_small_f32
// CHECK: attributes #

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func internal @__ocml_small_f32(%arg0: f32) -> f32 attributes {
    passthrough = ["alwaysinline"]
  } {
    %one = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %sum = llvm.fadd %arg0, %one : f32
    llvm.return %sum : f32
  }

  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr) {
    %value = llvm.load %arg0 : !llvm.ptr -> f32
    %c000 = llvm.call @__ocml_small_f32(%value) : (f32) -> f32
    %c001 = llvm.call @__ocml_small_f32(%c000) : (f32) -> f32
    %c002 = llvm.call @__ocml_small_f32(%c001) : (f32) -> f32
    %c003 = llvm.call @__ocml_small_f32(%c002) : (f32) -> f32
    %c004 = llvm.call @__ocml_small_f32(%c003) : (f32) -> f32
    %c005 = llvm.call @__ocml_small_f32(%c004) : (f32) -> f32
    %c006 = llvm.call @__ocml_small_f32(%c005) : (f32) -> f32
    %c007 = llvm.call @__ocml_small_f32(%c006) : (f32) -> f32
    %c008 = llvm.call @__ocml_small_f32(%c007) : (f32) -> f32
    %c009 = llvm.call @__ocml_small_f32(%c008) : (f32) -> f32
    %c010 = llvm.call @__ocml_small_f32(%c009) : (f32) -> f32
    %c011 = llvm.call @__ocml_small_f32(%c010) : (f32) -> f32
    %c012 = llvm.call @__ocml_small_f32(%c011) : (f32) -> f32
    %c013 = llvm.call @__ocml_small_f32(%c012) : (f32) -> f32
    %c014 = llvm.call @__ocml_small_f32(%c013) : (f32) -> f32
    %c015 = llvm.call @__ocml_small_f32(%c014) : (f32) -> f32
    %c016 = llvm.call @__ocml_small_f32(%c015) : (f32) -> f32
    %c017 = llvm.call @__ocml_small_f32(%c016) : (f32) -> f32
    %c018 = llvm.call @__ocml_small_f32(%c017) : (f32) -> f32
    %c019 = llvm.call @__ocml_small_f32(%c018) : (f32) -> f32
    %c020 = llvm.call @__ocml_small_f32(%c019) : (f32) -> f32
    %c021 = llvm.call @__ocml_small_f32(%c020) : (f32) -> f32
    %c022 = llvm.call @__ocml_small_f32(%c021) : (f32) -> f32
    %c023 = llvm.call @__ocml_small_f32(%c022) : (f32) -> f32
    %c024 = llvm.call @__ocml_small_f32(%c023) : (f32) -> f32
    %c025 = llvm.call @__ocml_small_f32(%c024) : (f32) -> f32
    %c026 = llvm.call @__ocml_small_f32(%c025) : (f32) -> f32
    %c027 = llvm.call @__ocml_small_f32(%c026) : (f32) -> f32
    %c028 = llvm.call @__ocml_small_f32(%c027) : (f32) -> f32
    %c029 = llvm.call @__ocml_small_f32(%c028) : (f32) -> f32
    %c030 = llvm.call @__ocml_small_f32(%c029) : (f32) -> f32
    %c031 = llvm.call @__ocml_small_f32(%c030) : (f32) -> f32
    %c032 = llvm.call @__ocml_small_f32(%c031) : (f32) -> f32
    %c033 = llvm.call @__ocml_small_f32(%c032) : (f32) -> f32
    %c034 = llvm.call @__ocml_small_f32(%c033) : (f32) -> f32
    %c035 = llvm.call @__ocml_small_f32(%c034) : (f32) -> f32
    %c036 = llvm.call @__ocml_small_f32(%c035) : (f32) -> f32
    %c037 = llvm.call @__ocml_small_f32(%c036) : (f32) -> f32
    %c038 = llvm.call @__ocml_small_f32(%c037) : (f32) -> f32
    %c039 = llvm.call @__ocml_small_f32(%c038) : (f32) -> f32
    %c040 = llvm.call @__ocml_small_f32(%c039) : (f32) -> f32
    %c041 = llvm.call @__ocml_small_f32(%c040) : (f32) -> f32
    %c042 = llvm.call @__ocml_small_f32(%c041) : (f32) -> f32
    %c043 = llvm.call @__ocml_small_f32(%c042) : (f32) -> f32
    %c044 = llvm.call @__ocml_small_f32(%c043) : (f32) -> f32
    %c045 = llvm.call @__ocml_small_f32(%c044) : (f32) -> f32
    %c046 = llvm.call @__ocml_small_f32(%c045) : (f32) -> f32
    %c047 = llvm.call @__ocml_small_f32(%c046) : (f32) -> f32
    %c048 = llvm.call @__ocml_small_f32(%c047) : (f32) -> f32
    %c049 = llvm.call @__ocml_small_f32(%c048) : (f32) -> f32
    %c050 = llvm.call @__ocml_small_f32(%c049) : (f32) -> f32
    %c051 = llvm.call @__ocml_small_f32(%c050) : (f32) -> f32
    %c052 = llvm.call @__ocml_small_f32(%c051) : (f32) -> f32
    %c053 = llvm.call @__ocml_small_f32(%c052) : (f32) -> f32
    %c054 = llvm.call @__ocml_small_f32(%c053) : (f32) -> f32
    %c055 = llvm.call @__ocml_small_f32(%c054) : (f32) -> f32
    %c056 = llvm.call @__ocml_small_f32(%c055) : (f32) -> f32
    %c057 = llvm.call @__ocml_small_f32(%c056) : (f32) -> f32
    %c058 = llvm.call @__ocml_small_f32(%c057) : (f32) -> f32
    %c059 = llvm.call @__ocml_small_f32(%c058) : (f32) -> f32
    %c060 = llvm.call @__ocml_small_f32(%c059) : (f32) -> f32
    %c061 = llvm.call @__ocml_small_f32(%c060) : (f32) -> f32
    %c062 = llvm.call @__ocml_small_f32(%c061) : (f32) -> f32
    %c063 = llvm.call @__ocml_small_f32(%c062) : (f32) -> f32
    %c064 = llvm.call @__ocml_small_f32(%c063) : (f32) -> f32
    %c065 = llvm.call @__ocml_small_f32(%c064) : (f32) -> f32
    %c066 = llvm.call @__ocml_small_f32(%c065) : (f32) -> f32
    %c067 = llvm.call @__ocml_small_f32(%c066) : (f32) -> f32
    %c068 = llvm.call @__ocml_small_f32(%c067) : (f32) -> f32
    %c069 = llvm.call @__ocml_small_f32(%c068) : (f32) -> f32
    %c070 = llvm.call @__ocml_small_f32(%c069) : (f32) -> f32
    %c071 = llvm.call @__ocml_small_f32(%c070) : (f32) -> f32
    %c072 = llvm.call @__ocml_small_f32(%c071) : (f32) -> f32
    %c073 = llvm.call @__ocml_small_f32(%c072) : (f32) -> f32
    %c074 = llvm.call @__ocml_small_f32(%c073) : (f32) -> f32
    %c075 = llvm.call @__ocml_small_f32(%c074) : (f32) -> f32
    %c076 = llvm.call @__ocml_small_f32(%c075) : (f32) -> f32
    %c077 = llvm.call @__ocml_small_f32(%c076) : (f32) -> f32
    %c078 = llvm.call @__ocml_small_f32(%c077) : (f32) -> f32
    %c079 = llvm.call @__ocml_small_f32(%c078) : (f32) -> f32
    %c080 = llvm.call @__ocml_small_f32(%c079) : (f32) -> f32
    %c081 = llvm.call @__ocml_small_f32(%c080) : (f32) -> f32
    %c082 = llvm.call @__ocml_small_f32(%c081) : (f32) -> f32
    %c083 = llvm.call @__ocml_small_f32(%c082) : (f32) -> f32
    %c084 = llvm.call @__ocml_small_f32(%c083) : (f32) -> f32
    %c085 = llvm.call @__ocml_small_f32(%c084) : (f32) -> f32
    %c086 = llvm.call @__ocml_small_f32(%c085) : (f32) -> f32
    %c087 = llvm.call @__ocml_small_f32(%c086) : (f32) -> f32
    %c088 = llvm.call @__ocml_small_f32(%c087) : (f32) -> f32
    %c089 = llvm.call @__ocml_small_f32(%c088) : (f32) -> f32
    %c090 = llvm.call @__ocml_small_f32(%c089) : (f32) -> f32
    %c091 = llvm.call @__ocml_small_f32(%c090) : (f32) -> f32
    %c092 = llvm.call @__ocml_small_f32(%c091) : (f32) -> f32
    %c093 = llvm.call @__ocml_small_f32(%c092) : (f32) -> f32
    %c094 = llvm.call @__ocml_small_f32(%c093) : (f32) -> f32
    %c095 = llvm.call @__ocml_small_f32(%c094) : (f32) -> f32
    %c096 = llvm.call @__ocml_small_f32(%c095) : (f32) -> f32
    %c097 = llvm.call @__ocml_small_f32(%c096) : (f32) -> f32
    %c098 = llvm.call @__ocml_small_f32(%c097) : (f32) -> f32
    %c099 = llvm.call @__ocml_small_f32(%c098) : (f32) -> f32
    %c100 = llvm.call @__ocml_small_f32(%c099) : (f32) -> f32
    %c101 = llvm.call @__ocml_small_f32(%c100) : (f32) -> f32
    %c102 = llvm.call @__ocml_small_f32(%c101) : (f32) -> f32
    %c103 = llvm.call @__ocml_small_f32(%c102) : (f32) -> f32
    %c104 = llvm.call @__ocml_small_f32(%c103) : (f32) -> f32
    %c105 = llvm.call @__ocml_small_f32(%c104) : (f32) -> f32
    %c106 = llvm.call @__ocml_small_f32(%c105) : (f32) -> f32
    %c107 = llvm.call @__ocml_small_f32(%c106) : (f32) -> f32
    %c108 = llvm.call @__ocml_small_f32(%c107) : (f32) -> f32
    %c109 = llvm.call @__ocml_small_f32(%c108) : (f32) -> f32
    %c110 = llvm.call @__ocml_small_f32(%c109) : (f32) -> f32
    %c111 = llvm.call @__ocml_small_f32(%c110) : (f32) -> f32
    %c112 = llvm.call @__ocml_small_f32(%c111) : (f32) -> f32
    %c113 = llvm.call @__ocml_small_f32(%c112) : (f32) -> f32
    %c114 = llvm.call @__ocml_small_f32(%c113) : (f32) -> f32
    %c115 = llvm.call @__ocml_small_f32(%c114) : (f32) -> f32
    %c116 = llvm.call @__ocml_small_f32(%c115) : (f32) -> f32
    %c117 = llvm.call @__ocml_small_f32(%c116) : (f32) -> f32
    %c118 = llvm.call @__ocml_small_f32(%c117) : (f32) -> f32
    %c119 = llvm.call @__ocml_small_f32(%c118) : (f32) -> f32
    %c120 = llvm.call @__ocml_small_f32(%c119) : (f32) -> f32
    %c121 = llvm.call @__ocml_small_f32(%c120) : (f32) -> f32
    %c122 = llvm.call @__ocml_small_f32(%c121) : (f32) -> f32
    %c123 = llvm.call @__ocml_small_f32(%c122) : (f32) -> f32
    %c124 = llvm.call @__ocml_small_f32(%c123) : (f32) -> f32
    %c125 = llvm.call @__ocml_small_f32(%c124) : (f32) -> f32
    %c126 = llvm.call @__ocml_small_f32(%c125) : (f32) -> f32
    %c127 = llvm.call @__ocml_small_f32(%c126) : (f32) -> f32
    llvm.store %c127, %arg0 : f32, !llvm.ptr
    llvm.return
  }
}
