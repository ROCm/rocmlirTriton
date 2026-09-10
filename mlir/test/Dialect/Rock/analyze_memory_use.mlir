// RUN: rocmlir-opt -rock-analyze-memory-use %s | FileCheck %s

// Note: the 64-bit index support is tested in large_tensor_detection

// CHECK-LABEL: @base_case
// CHECK-SAME: (%{{.*}}: tensor<16xf32> {llvm.align = 16 : i64, llvm.dereferenceable = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly, tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %{{.*}}: tensor<16xf32> {llvm.align = 16 : i64, llvm.dereferenceable = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.writeonly, tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %{{.*}}: index)
func.func @base_case(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>, %arg2: index) -> tensor<16xf32> attributes {rock.kernel} {
  %v = rock.blockwise_load %arg0 {cacheModifier = #rock<CacheModifier none>} : tensor<16xf32> -> tensor<16xf32>
  %out = rock.blockwise_store %v -> %arg1 by set
    : tensor<16xf32> -> tensor<16xf32> -> tensor<16xf32>
  return %out : tensor<16xf32>
}

// arg0 is loaded → readonly; arg1 is atomic-stored → neither readonly nor writeonly
// CHECK-LABEL: @atomic_case
// CHECK-SAME: (%{{.*}}: tensor<16xf32> {{{.*}}llvm.readonly{{.*}}}, %{{.*}}: tensor<16xf32> {llvm.align = 16 : i64, llvm.dereferenceable = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
func.func @atomic_case(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>, %arg2: index) -> tensor<16xf32> attributes {rock.kernel} {
  %v = rock.blockwise_load %arg0 {cacheModifier = #rock<CacheModifier none>} : tensor<16xf32> -> tensor<16xf32>
  %out = rock.blockwise_store %v -> %arg1 by atomic_add
    : tensor<16xf32> -> tensor<16xf32> -> tensor<16xf32>
  return %out : tensor<16xf32>
}

// CHECK-LABEL: @dead_arg
// CHECK-SAME: llvm.readnone
func.func @dead_arg(%arg0: tensor<16xf32>) attributes {rock.kernel} {
  return
}

// CHECK-LABEL: @dynamic_shape
// CHECK-NOT: llvm.dereferenceable
// CHECK-NOT: tt.pointer_range
func.func @dynamic_shape(%arg0: tensor<?xf32>) -> tensor<?xf32> attributes {rock.kernel} {
  %v = rock.blockwise_load %arg0 {cacheModifier = #rock<CacheModifier none>} : tensor<?xf32> -> tensor<?xf32>
  %out = rock.blockwise_store %v -> %arg0 by atomic_add
    : tensor<?xf32> -> tensor<?xf32> -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// f4 type with odd element count: dereferenceable = ceil(3*4/8) = 2
// CHECK-LABEL: @f4_odd_length
// CHECK-SAME: (%{{.*}}: tensor<3xf4E2M1FN> {llvm.align = 16 : i64, llvm.dereferenceable = 2 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readnone, tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
func.func @f4_odd_length(%arg0: tensor<3xf4E2M1FN>) attributes {rock.kernel} {
  return
}

// An arg that is only loaded from should still be readonly even when another
// arg is only stored to (i.e. no cross-contamination between args).
// CHECK-LABEL: @separate_read_write
// CHECK-SAME: (%{{.*}}: tensor<16xf32> {{{.*}}llvm.readonly{{.*}}}, %{{.*}}: tensor<16xf32> {{{.*}}llvm.writeonly{{.*}}}
func.func @separate_read_write(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %v = rock.blockwise_load %arg0 {cacheModifier = #rock<CacheModifier none>} : tensor<16xf32> -> tensor<16xf32>
  %out = rock.blockwise_store %v -> %arg1 by set
    : tensor<16xf32> -> tensor<16xf32> -> tensor<16xf32>
  return %out : tensor<16xf32>
}

// CHECK-LABEL: @non_kernel_skipped
// CHECK-NOT: llvm.noalias
func.func @non_kernel_skipped(%arg0: tensor<16xf32>) {
  return
}

// A transformed dense compiler constant is loaded from internal storage and
// must not be treated as a kernel argument. The only argument remains
// write-only.
// CHECK-LABEL: @dense_constant
// CHECK-SAME: (%{{.*}}: tensor<4xf32> {{{.*}}llvm.writeonly{{.*}}})
func.func @dense_constant(%dest: tensor<4xf32>) -> tensor<4xf32> attributes {rock.kernel} {
  %values = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %view = rock.transform %values by <affine_map<(d0) -> (d0)> by [<PassThrough ["element"] at [0] -> ["element"] at [0]>] bounds = [4] -> [4]> : tensor<4xf32> to tensor<4xf32>
  %loaded = rock.blockwise_load %view {cacheModifier = #rock<CacheModifier none>} : tensor<4xf32> -> tensor<4xf32>
  %result = rock.blockwise_store %loaded -> %dest by set
    : tensor<4xf32> -> tensor<4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}
