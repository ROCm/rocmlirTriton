// How rocmlir-gen fills in rock.num_cu when --num_cu is absent. A live count
// cannot be checked, since it is whatever the host reports, so every run below
// either supplies a count or hides the devices and pins the fallback.

// 1. An explicit count always wins, whatever a device would have reported.

// RUN: rocmlir-gen --arch gfx942 --num_cu 7 --operation gemm -t f16 -g 1 -m 64 -k 64 -n 64 \
// RUN: | FileCheck %s --check-prefix=PINNED

// PINNED: rock.num_cu = 7 : i64

// 2. With no device visible, which is what a GPU-less compile host looks like,
//    the query yields nothing and the per-arch default has to hold.

// RUN: env HIP_VISIBLE_DEVICES=-1 \
// RUN:   rocmlir-gen --arch gfx942 --operation gemm -t f16 -g 1 -m 64 -k 64 -n 64 \
// RUN: | FileCheck %s --check-prefix=CDNA3

// CDNA3: rock.num_cu = 20 : i64

// 3. gfx950 is the one architecture whose assumed count is not its floor: the
//    floor is CPX's single 32-CU partition, while an unpartitioned card has all
//    256, so getDefaultNumCU has to hand back the latter. Devices are hidden
//    here too, since a gfx950 host would answer for itself.

// RUN: env HIP_VISIBLE_DEVICES=-1 \
// RUN:   rocmlir-gen --arch gfx950 --operation gemm -t f16 -g 1 -m 64 -k 64 -n 64 \
// RUN: | FileCheck %s --check-prefix=CDNA4

// CDNA4: rock.num_cu = 256 : i64

// 4. A count below the family's floor survives lowering, because a partitioned
//    gfx950 legitimately reports one. getNumCUOnFunc used to reject anything
//    under getMinNumCU, which rejected all three of DPX, QPX and CPX.

// RUN: rocmlir-gen --arch gfx950 --num_cu 32 --operation gemm -t f16 -g 1 -m 64 -k 64 -n 64 \
// RUN: | rocmlir-driver -kernel-pipeline=gpu -arch gfx950 \
// RUN: | FileCheck %s --check-prefix=PARTITIONED

// PARTITIONED: rock.num_cu = 32 : i64

// 5. The count is not decoration, it reaches codegen. GridLayoutEmitter derives
//    the grid group size from numCU/numChiplets whenever the tuned gridGroupSize
//    is zero, which is the default, so an assumed count builds a different
//    kernel than a measured one. Both are pinned below: the group size lands at
//    4 for an assumed CDNA3 and at 14 for the 304 units an unpartitioned MI300X
//    reports, and it shows up in the grid arithmetic both on its own and times
//    nBlocks. Counts are explicit so that this holds on any host.

// RUN: rocmlir-gen --arch gfx942 --num_cu 20 --num_chiplets 8 --operation gemm \
// RUN:   -t f16 -out_datatype f32 -g 1 -m 4096 -k 4096 -n 4096 -transA=False -transB=False \
// RUN: | rocmlir-driver -kernel-pipeline=gpu -arch gfx942 \
// RUN: | FileCheck %s --check-prefix=ASSUMED

// ASSUMED-DAG: arith.constant 4 : i32
// ASSUMED-DAG: arith.constant 256 : i32
// ASSUMED-NOT: arith.constant 14 : i32
// ASSUMED-NOT: arith.constant 896 : i32

// RUN: rocmlir-gen --arch gfx942 --num_cu 304 --num_chiplets 8 --operation gemm \
// RUN:   -t f16 -out_datatype f32 -g 1 -m 4096 -k 4096 -n 4096 -transA=False -transB=False \
// RUN: | rocmlir-driver -kernel-pipeline=gpu -arch gfx942 \
// RUN: | FileCheck %s --check-prefix=MEASURED

// MEASURED-DAG: arith.constant 14 : i32
// MEASURED-DAG: arith.constant 896 : i32
// MEASURED-NOT: arith.constant 4 : i32
// MEASURED-NOT: arith.constant 256 : i32
