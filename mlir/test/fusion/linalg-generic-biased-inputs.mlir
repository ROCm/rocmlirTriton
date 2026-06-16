// RUN: rocmlir-opt -rock-lower-blockwise-to-ptr -rock-preserve-masked-load-semantics %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#transform_map = #rock.transform_map<#map by [<Pad{0, 12} ["xPad"] at [0] -> ["x"] at [0]>] bounds = [16] -> [4]>

// CHECK-LABEL: @must_reapply_padding
// CHECK: rock.blockwise_load_ptr
// CHECK: %[[biased:.+]] = arith.addf
// CHECK: arith.select {{.*}}, %[[biased]],
// CHECK: rock.blockwise_store_ptr
// CHECK: return
func.func @must_reapply_padding(%arg0: tensor<4xf16>, %dst: tensor<16xf16>) -> tensor<16xf16> attributes {rock.kernel} {
  %cst = arith.constant dense<4.0> : tensor<16xf16>
  %padded = rock.transform %arg0 by #transform_map : tensor<4xf16> to tensor<16xf16>
  %loaded = rock.blockwise_load %padded {cacheModifier = #rock<CacheModifier none>} : tensor<16xf16> -> tensor<16xf16>
  %biased = arith.addf %loaded, %cst : tensor<16xf16>
  %stored = rock.blockwise_store %biased -> %dst by set : tensor<16xf16> -> tensor<16xf16> -> tensor<16xf16>
  return %stored : tensor<16xf16>
}

// CHECK-LABEL: @doesnt_reapply_padding
// CHECK: rock.blockwise_load_ptr
// CHECK: arith.mulf
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr
// CHECK: return
func.func @doesnt_reapply_padding(%arg0: tensor<4xf16>, %dst: tensor<16xf16>) -> tensor<16xf16> attributes {rock.kernel} {
  %cst = arith.constant dense<4.0> : tensor<16xf16>
  %padded = rock.transform %arg0 by #transform_map : tensor<4xf16> to tensor<16xf16>
  %loaded = rock.blockwise_load %padded {cacheModifier = #rock<CacheModifier none>} : tensor<16xf16> -> tensor<16xf16>
  %scaled = arith.mulf %loaded, %cst : tensor<16xf16>
  %stored = rock.blockwise_store %scaled -> %dst by set : tensor<16xf16> -> tensor<16xf16> -> tensor<16xf16>
  return %stored : tensor<16xf16>
}
