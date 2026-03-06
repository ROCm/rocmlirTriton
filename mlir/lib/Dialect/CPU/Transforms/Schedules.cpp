//===- Schedules.cpp - CPU lowering transform schedules -------------------===//
//
// Copyright 2026 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file contains the transform schedules used for CPU lowering.
// The schedules are MLIR transform dialect modules that define the
// optimization and lowering pipeline.
//
//===----------------------------------------------------------------------===//

#include "Schedules.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cpu-schedules"

using namespace mlir;
using namespace mlir::cpu;

//===----------------------------------------------------------------------===//
// Transform Schedule Definitions (MLIR text format)
//===----------------------------------------------------------------------===//

namespace {

/// Pre-sequence: canonicalize + CSE
/// Applied before each major phase to clean up the IR.
constexpr const char *kPreSequence = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.apply_registered_pass "canonicalize" to %1 : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %2 : !transform.any_op
    transform.yield 
  }
}
)mlir";

/// Post-sequence: LICM + hoisting + canonicalize
/// Applied after each major phase to optimize loops.
constexpr const char *kPostSequence = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match interface{LoopLikeInterface} in %1 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %2 : !transform.any_op
    
    %func = transform.structured.match ops{["func.func"]} in %1 : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.hoist_redundant_vector_transfers %func : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.hoist_redundant_vector_broadcasts %3 : (!transform.any_op) -> !transform.any_op
    %5 = transform.apply_registered_pass "canonicalize" to %4 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}
)mlir";

/// Optimization sequence: tiling for CPU cache efficiency.
/// This sequence tiles the matmul and applies canonicalization.
/// It ends before bufferization.
constexpr const char *kOptimizationSequence = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    // TODO: Get rid of unit dims (for batch mamtul)
    %0 = transform.structured.match attributes {rock.matmul} in %arg0 : (!transform.any_op) -> !transform.any_op

    %transformed, %loops:3 = transform.structured.fuse %0 tile_sizes [1, 256, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %xxx, %loop = transform.structured.tile_using_for %transformed tile_sizes [0, 0, 0, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %microkernel, %microkernel_loops:3 = transform.structured.tile_using_for %xxx tile_sizes [0, 8, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // TODO: We will probably need padding and/or packing here.

    %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.apply_registered_pass "canonicalize" to %1 : (!transform.any_op) -> !transform.any_op
    %3 = transform.apply_registered_pass "lower-affine" to %2 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}
)mlir";

/// Vectorization sequence: vectorize linalg ops.
constexpr const char *kVectorizationSequence = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %0 = transform.structured.match attributes {rock.matmul} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 {vectorize_nd_extract, vectorize_padding} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
)mlir";

/// Lowering sequence: bufferization + vector lowering + LLVM conversion.
/// This sequence starts with bufferization and lowers to LLVM dialect.
constexpr const char *kLoweringSequence = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {

    %10 = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %arg0 {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op

    // Lowering to LLVM

    %func = transform.structured.match ops{["func.func"]} in %10 : (!transform.any_op) -> !transform.any_op
    // Bufferization is quite dumb, so it will probably insert unnecesary memref copies.
    // We use CSE + canonicalize to remove them.
    %3 = transform.apply_registered_pass "cse" to %func : (!transform.any_op) -> !transform.any_op
    %4 = transform.apply_registered_pass "canonicalize" to %3 : (!transform.any_op) -> !transform.any_op

    // Add these after bufferization to hoist allocations out of loops
    %b10 = transform.apply_registered_pass "buffer-hoisting" to %4 : (!transform.any_op) -> !transform.any_op
    %c10 = transform.apply_registered_pass "buffer-loop-hoisting" to %b10 : (!transform.any_op) -> !transform.any_op
    // %d10 = transform.apply_registered_pass "buffer-deallocation" to %c10 : (!transform.any_op) -> !transform.any_op


    // Now time to lower vectors.
    %fff = transform.structured.match ops{["func.func"]} in %c10 : (!transform.any_op) -> !transform.any_op

    // TODO: group these lower-level controls into various properly named vector
    // lowering TD macros.
    transform.apply_patterns to %fff {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.transfer_permutation_patterns
      // transform.apply_patterns.vector.reorder_and_expand_multi_reduction_dims lowering_strategy = "innerparallel"
      // transform.apply_patterns.vector.multi_reduction_flattening lowering_strategy = "innerparallel"
      // transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = "innerparallel"
      transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "linalg-copy"
      transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 4 full_unroll = true
      transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
      transform.apply_patterns.vector.lower_shape_cast
      transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
    } : !transform.any_op

    // Now time to lower to LLVM
    %f = transform.apply_registered_pass "convert-vector-to-scf" to %fff : (!transform.any_op) -> !transform.any_op
    %f2 = transform.apply_registered_pass "convert-linalg-to-loops" to %f : (!transform.any_op) -> !transform.any_op

    // IMPORTANT: convert-vector-to-scf creates new allocations for vector transfers.
    // We must run buffer-loop-hoisting again to move these out of loops,
    // otherwise we get stack overflow from allocas inside loops.
    %fh1 = transform.apply_registered_pass "buffer-loop-hoisting" to %f2 : (!transform.any_op) -> !transform.any_op
    %fh2 = transform.apply_registered_pass "promote-buffers-to-stack" to %fh1 : (!transform.any_op) -> !transform.any_op

    %f3 = transform.apply_registered_pass "convert-scf-to-cf" to %fh2 : (!transform.any_op) -> !transform.any_op
    %f4 = transform.apply_registered_pass "expand-strided-metadata" to %f3 : (!transform.any_op) -> !transform.any_op
    %f5 = transform.apply_registered_pass "lower-affine" to %f4 : (!transform.any_op) -> !transform.any_op
    %f5b = transform.apply_registered_pass "convert-vector-to-llvm" with options = {"reassociate-fp-reductions" = false} to %f5 : (!transform.any_op) -> !transform.any_op

    transform.apply_conversion_patterns to %f5b {
      transform.apply_conversion_patterns.dialect_to_llvm "math"
      transform.apply_conversion_patterns.vector.vector_to_llvm
      transform.apply_conversion_patterns.dialect_to_llvm "memref"
      // transform.apply_conversion_patterns.func.func_to_llvm
      transform.apply_conversion_patterns.dialect_to_llvm "index"
      transform.apply_conversion_patterns.dialect_to_llvm "arith"
      transform.apply_conversion_patterns.dialect_to_llvm "cf"
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
        {index_bitwidth = 64,
        use_bare_ptr = false,
        use_bare_ptr_memref_call_conv = false,
        use_opaque_pointers = true}
    } {
      legal_dialects = ["llvm"],
      partial_conversion
    } : !transform.any_op

    // Need to rematch here because:
    //   1. applying reconcile-unrealized-casts on the whole module yields the
    //      transform applies to transform, when called from a named sequence, at
    //      this time.
    //   2. apply_conversion patterns consumes the func but does not produce
    //      a new llvm.func.
    %f6 = transform.structured.match ops{["llvm.func"]} in %10
      : (!transform.any_op) -> !transform.any_op
    %f7 = transform.apply_registered_pass "reconcile-unrealized-casts" to %f6
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
)mlir";

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------------===//

void cpu::registerScheduleDialectExtensions(DialectRegistry &registry) {
  // Register transform dialect extensions needed for parsing the transform
  // sequences. This must be done in getDependentDialects() to avoid
  // thread-safety issues. The extensions provide ops like
  // transform.structured.match, transform.bufferization.one_shot_bufferize.
  linalg::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  vector::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);

  // Register ValueBoundsOpInterface for dialects, required by
  // transform.structured.tile_using_for and other tiling transforms
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);

  // Register BufferizableOpInterface and SubsetOpInterface for vector dialect,
  // required by one_shot_bufferize when vector ops are present
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerSubsetOpInterfaceExternalModels(registry);

  // Register AllocationOpInterface for memref dialect,
  // required by buffer-hoisting and buffer-loop-hoisting passes
  memref::registerAllocationOpInterfaceExternalModels(registry);
}

FailureOr<TransformSchedules> cpu::createTransformSchedules(MLIRContext *ctx) {
  TransformSchedules schedules;

  // Pre-sequence (canonicalize + cse)
  schedules.preModule = parseSourceString<ModuleOp>(kPreSequence, ctx);
  if (!schedules.preModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to parse pre transform sequence for CPU verifier";
    return failure();
  }

  // Post-sequence (LICM + hoisting)
  schedules.postModule = parseSourceString<ModuleOp>(kPostSequence, ctx);
  if (!schedules.postModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to parse post transform sequence for CPU verifier";
    return failure();
  }

  // Optimization sequence (tiling)
  schedules.optimizationModule =
      parseSourceString<ModuleOp>(kOptimizationSequence, ctx);
  if (!schedules.optimizationModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to parse optimization transform sequence for CPU verifier";
    return failure();
  }

  // Vectorization sequence
  schedules.vectorizationModule =
      parseSourceString<ModuleOp>(kVectorizationSequence, ctx);
  if (!schedules.vectorizationModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to parse vectorization transform sequence for CPU verifier";
    return failure();
  }

  // Lowering sequence (bufferization + LLVM)
  schedules.loweringModule =
      parseSourceString<ModuleOp>(kLoweringSequence, ctx);
  if (!schedules.loweringModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to parse lowering transform sequence for CPU verifier";
    return failure();
  }

  return schedules;
}

LogicalResult cpu::applyTransformSequence(ModuleOp targetModule,
                                          ModuleOp transformModule,
                                          StringRef sequenceName) {
  // Find the transform entry point
  transform::TransformOpInterface entryPoint =
      transform::detail::findTransformEntryPoint(targetModule, transformModule);
  if (!entryPoint) {
    return targetModule.emitError("Could not find transform entry point for ")
           << sequenceName;
  }

  // Apply the transform sequence to the target module
  transform::TransformOptions options;
  if (failed(transform::applyTransformNamedSequence(targetModule, entryPoint,
                                                    transformModule, options))) {
    return targetModule.emitError("Transform interpreter failed for ")
           << sequenceName;
  }

  return success();
}
