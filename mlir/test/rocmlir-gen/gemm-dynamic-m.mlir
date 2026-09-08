// RUN: rocmlir-gen -ph -fut dot_dynamic_m -m 256 %s | FileCheck %s
// RUN: not rocmlir-gen -ph -fut dot_dynamic_m %s 2>&1 | FileCheck %s --check-prefix=NO-M

// A harness has to commit to one M in order to allocate, so its buffers come
// out static even though the kernel stays polymorphic. A is M x K and the
// output is M x N, both flattened, so with -m 256 they hold 256 * 64 elements
// while B keeps its own size.

// CHECK-LABEL: func.func @main
// CHECK: memref.alloc() : memref<16384xf32>
// CHECK: memref.alloc() : memref<4096xf32>

// The kernel is still declared with an unknown extent, so the concrete buffers
// are reconciled with it on the memref, before they become tensors: a
// tensor.cast here would keep bufferization from folding back to the device
// buffer and the launch would pass the GPU a host pointer.

// CHECK-LABEL: func.func @dot_dynamic_m_gpu
// CHECK: %[[GPU_A:.*]] = gpu.alloc () : memref<16384xf32>
// CHECK: %[[CAST_A:.*]] = memref.cast %[[GPU_A]] : memref<16384xf32> to memref<?xf32>
// CHECK: bufferization.to_tensor %[[CAST_A]]
// CHECK: call @dot_dynamic_m(

// NO-M: M dimension is dynamic, so -m is required

#map = affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
#map1 = affine_map<(d0) -> (0, d0 floordiv 64, d0 mod 64)>
#transform_map = #rock.transform_map<#map by [<Unmerge{?, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, ?, 64] -> [?]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]>
#transform_map2 = #rock.transform_map<#map1 by [<Merge{1, ?, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [?] -> [1, ?, 64]>
module attributes {rock.arch = "gfx942"} {
  func.func @dot_dynamic_m(%arg0: tensor<?xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> attributes {rock.kernel} {
    %0 = rock.transform %arg0 by #transform_map : tensor<?xf32> to tensor<1x?x64xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<4096xf32> to tensor<1x64x64xf32>
    %2 = rock.gemm %0 * %1 : tensor<1x?x64xf32> * tensor<1x64x64xf32> -> tensor<1x?x64xf32>
    %3 = rock.transform %2 by #transform_map2 : tensor<1x?x64xf32> to tensor<?xf32>
    %4 = rock.store %3 to %arg2 by set : tensor<?xf32> -> tensor<?xf32> to tensor<?xf32>
    return %4 : tensor<?xf32>
  }
}
