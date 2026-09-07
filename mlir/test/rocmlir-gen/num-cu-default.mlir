// How rocmlir-gen fills in rock.num_cu when --num_cu is absent. The live count
// itself cannot be checked, since it is whatever the host reports, so each run
// below arranges for the query to yield nothing and pins the fallback instead.

// 1. An explicit count always wins, whatever a device would have reported.

// RUN: rocmlir-gen --arch gfx942 --num_cu 7 --operation gemm -t f16 -g 1 -m 64 -k 64 -n 64 \
// RUN: | FileCheck %s --check-prefix=PINNED

// PINNED: rock.num_cu = 7 : i64

// 2. ROCMLIR_GEN_NO_NATIVE_CU_QUERY, which mlir/test/lit.cfg.py exports for the
//    whole suite, skips the query outright. Without it the suite's output would
//    depend on the host's GPU and every invocation would initialize HIP.

// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -g 1 -m 64 -k 64 -n 64 \
// RUN: | FileCheck %s --check-prefix=CDNA3

// 3. With the query enabled but no device visible -- a compile host without a
//    GPU -- the fallback has to hold, which is the path this run pins.

// RUN: env -u ROCMLIR_GEN_NO_NATIVE_CU_QUERY HIP_VISIBLE_DEVICES=-1 \
// RUN:   rocmlir-gen --arch gfx942 --operation gemm -t f16 -g 1 -m 64 -k 64 -n 64 \
// RUN: | FileCheck %s --check-prefix=CDNA3

// CDNA3: rock.num_cu = 20 : i64

// 4. gfx950 is the one architecture whose assumed count is not its floor: the
//    floor is CPX's single 32-CU partition, while an unpartitioned card has all
//    256, so getDefaultNumCU has to hand back the latter.

// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -g 1 -m 64 -k 64 -n 64 \
// RUN: | FileCheck %s --check-prefix=CDNA4

// CDNA4: rock.num_cu = 256 : i64

// 5. A count below the family's floor survives lowering, because a partitioned
//    gfx950 legitimately reports one. getNumCUOnFunc used to reject anything
//    under getMinNumCU, which rejected all three of DPX, QPX and CPX.

// RUN: rocmlir-gen --arch gfx950 --num_cu 32 --operation gemm -t f16 -g 1 -m 64 -k 64 -n 64 \
// RUN: | rocmlir-driver -kernel-pipeline=gpu -arch gfx950 \
// RUN: | FileCheck %s --check-prefix=PARTITIONED

// PARTITIONED: rock.num_cu = 32 : i64
