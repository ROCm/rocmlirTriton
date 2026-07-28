// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %S/nightly-misc-e2e/mixr-attention/f16/mixr-attention-kvcache-clipped.mlir | FileCheck %s

// Verify the MIGraphX high-level pipeline lowers migraphx.clip to the pattern
// recognized by TosaToRock and specializes the KV-cache mask.
// CHECK-LABEL: func.func @mlir_attention
// CHECK: %[[CLAMPED:.*]] = arith.maxsi
// CHECK: %[[CLIPPED:.*]] = arith.minsi %[[CLAMPED]]
// CHECK: rock.attention
// CHECK: currentSeqLen = (%[[CLIPPED]] : tensor<2xi32>)
