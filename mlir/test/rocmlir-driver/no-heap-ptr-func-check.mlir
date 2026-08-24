// Verifies that the compiled code object never carries the device-heap implicit
// kernarg, so the HIP runtime never enqueues the one-time
// `__amd_rocclr_initHeap` setup kernel at module load.

// RUN: rocmlir-gen -operation gemm -t f16 -out_datatype f32 \
// RUN:   --arch gfx950 --num_cu 256 \
// RUN:   -g 1 -m 64 -k 64 -n 64 -transA=False -transB=False \
// RUN:   --perf_config="gemm:v4:16,16,16,1,1,2,16,1,2,0,0,-1,-1,-1,-1,-1,-1" \
// RUN:   | rocmlir-driver -c \
// RUN:   | FileCheck %s --implicit-check-not=hidden_heap_v1 --implicit-check-not=__amd_rocclr_initHeap

// CHECK-DAG matches in any order, so we don't depend on the relative position
// of the gpu.binary kernel list, the embedded HSA metadata, and the llvm.func.

// The compiled code object lists the expected kernel.
// CHECK-DAG: kernels = <[#gpu.kernel_metadata<"rock_gemm"

// The kernel carries the attribute that drops the device-heap implicit arg.
// CHECK-DAG: amdgpu-no-heap-ptr

// The HSA kernarg metadata is present and scanned (other hidden kernargs remain),
// while hidden_heap_v1 / __amd_rocclr_initHeap are rejected everywhere by the
// --implicit-check-not flags on the RUN line.
// CHECK-DAG: hidden_block_count_x
