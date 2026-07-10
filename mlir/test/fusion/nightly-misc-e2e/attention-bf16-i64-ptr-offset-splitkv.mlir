// Regression for 32-bit pointer-offset overflow in rocMLIR-generated attention
// kernels. This bf16 problem is large enough (and split_kv=32 pads/replicates
// the index domain) that the linearized element offset exceeds INT32_MAX. Before
// this branch the offset arithmetic feeding tt.addptr was emitted in i32, so the
// (nsw) 32-bit add overflowed and the kernel dereferenced out-of-range addresses,
// aborting at runtime with hipErrorIllegalAddress. The fix widens the base
// placeholder and offset arithmetic to i64 when 64-bit indexing is required, so
// address computation no longer overflows.
//
// Full verification run (-pv --pv-f64 compares against an f64 reference). Slow,
// so this lives under nightly-misc-e2e and only runs in the nightly CI stage
// (ROCMLIR_DRIVER_E2E_TEST_ENABLED / enable_rock_driver_e2e_test). Arch-independent:
// uses %arch and lets rocmlir-gen infer the hardware defaults.

// RUN: rocmlir-gen --arch %arch -operation attention -t bf16 -g 7 -seq_len_q 556 -seq_len_k 464 -num_heads_q 128 -num_heads_kv 8 -head_dim_qk 232 -head_dim_v 203 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=True -transV=False -transO=False -causal=False -return_lse=True -split_kv=32 --perf_config=attn:v3:32,64,128,1,1,2,16,1,2,1,8,-1,-1,-1,-1,-1 -pv --pv-f64 \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]

// Also verify at the IR level that the fix is in effect: the output tensor has
// 3236151296 (> INT32_MAX) elements, so TransformsToPointerArith must extract
// its base pointer as i64 and emit the store-offset arithmetic in i64 rather
// than i32 (which is what used to overflow).

// RUN: rocmlir-gen --arch %arch -operation attention -t bf16 -g 7 -seq_len_q 556 -seq_len_k 464 -num_heads_q 128 -num_heads_kv 8 -head_dim_qk 232 -head_dim_v 203 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=True -transV=False -transO=False -causal=False -return_lse=True -split_kv=32 --perf_config=attn:v3:32,64,128,1,1,2,16,1,2,1,8,-1,-1,-1,-1,-1 -pv --pv-f64 \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c --mlir-print-ir-after=rock-transforms-to-pointer-arith -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=I64

// I64: rock.extract_ptr %{{.*}} : tensor<3236151296xbf16> -> i64
// I64: rock.blockwise_store_ptr {{.*}} -> tensor<{{.*}}xi64>(
