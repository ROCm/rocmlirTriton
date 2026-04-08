// Verifies that the BakeKernelLaunchParams pass causes the AMDGPU backend to
// emit a non-zero .amdhsa_group_segment_fixed_size in the kernel descriptor,
// proving that LDS is allocated statically in the binary rather than passed
// dynamically at launch time.

// RUN: rocmlir-gen --arch gfx90a --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX90A
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX942

// The fixed size must be greater than zero — this means LDS is baked into the
// binary (static), not supplied at launch (dynamic).
// GFX90A: .amdhsa_group_segment_fixed_size {{[1-9][0-9]*}}
// GFX942: .amdhsa_group_segment_fixed_size {{[1-9][0-9]*}}
