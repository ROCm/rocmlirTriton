//===-- Passes.td - Rock pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_PASSES
#define MLIR_DIALECT_ROCK_PASSES

include "mlir/Pass/PassBase.td"

def RockViewToTransformPass : Pass<"rock-view-to-transform", "::mlir::func::FuncOp"> {
  let summary = "convert tensor view ops to rock.transform";
  let dependentDialects = ["rock::RockDialect", "tensor::TensorDialect",
    "tosa::TosaDialect"];
}

def RockDetectFlashDecodingPass
    : Pass<"rock-detect-flash-decoding", "::mlir::func::FuncOp"> {
  let summary = "Detect flash decoding (splitKV > 1) from rock.transform "
                "operations and update AttentionOps";
  let description = [{
    This pass analyzes rock.transform operations to detect when flash decoding
    (splitKV > 1) is being used. It identifies broadcast patterns in input
    tensors that indicate the splitKV factor, removes those broadcasts (since
    the attention implementation handles splitting internally), and updates the
    AttentionOp's splitKV attribute accordingly.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockConvToGemmPass : Pass<"rock-conv-to-gemm", "::mlir::func::FuncOp"> {
  let summary = "expand convolution into coordinate transformations and gemm. "
                "Additionally, it also convert rock.conv_elementwise_gemm into "
                "rock.gemm_elementwise_gemm";
  let dependentDialects = ["rock::RockDialect", "arith::ArithDialect",
                           "scf::SCFDialect", "gpu::GPUDialect"];
}

def RockAffixTuningParametersPass : Pass<"rock-affix-params", "::mlir::func::FuncOp"> {
  let summary = "populate tuning parameters for gemm and convolution ops";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockGemmToGridwisePass : Pass<"rock-gemm-to-gridwise", "::mlir::func::FuncOp"> {
  let summary = "convert gemm to gridwise gemm, adding padding";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "linalg::LinalgDialect", "arith::ArithDialect"];
}

def RockFoldBroadcastPass: Pass<"rock-fold-broadcast", "::mlir::func::FuncOp"> {
  let summary = "Convert Gemm([g, M, K], [broadcast{1}, K, N]) to Gemm([g*M, K], [K, N])";
  let dependentDialects = ["rock::RockDialect"];
}

def RockLowerReducePass : Pass<"rock-lower-reduce", "::mlir::func::FuncOp"> {
  let summary =
      "Lower rock.reduce into a Broadcast transform on the store destination";
  let description = [{
    This pass converts rock.reduce ops into an equivalent representation using
    a Broadcast transform on the rock.store destination. For sum reductions,
    this means:
    1. Bypassing the rock.reduce (passing its input directly to rock.store)
    2. Adding a Broadcast transform on the reduced axis of the store destination
    3. Changing the store method from `set` to `atomic_add`

    This allows the existing blockwise store pipeline to handle reductions
    without a dedicated reduction lowering.

    Example (reduce sum on axis=1):
      Before:
        %fused : tensor<1x100x100xf16>
        %reduced = rock.reduce sum %fused {axis = 1}
        %result = rock.store %reduced to %dest by set

      After:
        %fused : tensor<1x100x100xf16>
        %dest_bc = rock.transform %dest by <Broadcast{1} on axis 1>
        %result = rock.store %fused to %dest_bc by atomic_add
  }];
  let dependentDialects = ["rock::RockDialect", "arith::ArithDialect"];
}

def RockRegularizeOutputPass
    : Pass<"rock-regularize-output", "::mlir::func::FuncOp"> {
  let summary = "Move output fusions to operate directly in gemm output space";
  let description = [{
    This pass finds rock.transform ops between a FusionRoot (e.g., rock.gemm)
    result and output fusion ops (arith/math element-wise ops), then rewrites
    the IR so that fusions happen directly after the gemm. The intervening
    transforms are pushed past the fusions to sit between the last fusion and
    rock.store.

    The end goal of the pass is to make sure there are no rock.transform ops 
    between the FusionRoot and rock.store, there can only be fusion ops between
    FusionRoot and rock.store. Because of this, the pass inverts the 
    transforms applied to the fusion ops, thus avoiding the transformation of the 
    gemm output. This keeps all fusion computations in a canonical shape matching 
    the gemm output.

    Extra operands (non-gemm-chain inputs to the fusions) are handled by kind:
    - Splat arith.constant: recreated with the gemm-output type.
    - Other tensors: the inverse of the transform chain is applied.
    The pass fails if a non-constant extra operand requires a non-invertible
    transform (e.g., AddDim with length > 1).
  }];
  let dependentDialects = ["rock::RockDialect", "arith::ArithDialect"];
}

def RockGridwiseGemmToBlockwisePass : Pass<"rock-gridwise-gemm-to-blockwise", "::mlir::func::FuncOp"> {
  let summary = "expand gridwise gemm into blockwise copy, blockwise gemm, and threadwise copy";
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect",
                           "scf::SCFDialect", "triton::TritonDialect"];
}

def RockGridwiseAttnToBlockwisePass
    : Pass<"rock-gridwise-attn-to-blockwise", "::mlir::func::FuncOp"> {
  let summary = "expand gridwise attention into blockwise copy, blockwise "
                "attention, and threadwise copy";
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect",
                           "scf::SCFDialect", "math::MathDialect",
                           "triton::TritonDialect"];
}

def RockFusionSplitkRegularizationPass
    : Pass<"rock-fusion-splitk-regularization", "::mlir::func::FuncOp"> {
  let summary = "This pass modifies fusion ops for split-k fusions. It converts any arith.addf/arith.subf gemmOut, other to arith.addf gemmOut, other/splitkFactor";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockLegalizeFloatTypesPass
    : Pass<"rock-legalize-float-types", "::mlir::func::FuncOp"> {
  let summary = "Legalize float types not in Triton's TT_Float to integers";
  let description = [{
    Triton's tt.load only accepts TT_Float types. Float types outside that set
    (e.g. f8E8M0FNU, f4E2M1FN) are bitcast to integer types of the same bit
    width (i8, i4) so downstream TTIR lowering works.
    For kernel functions, replaces all non-TT_Float element types with
    matching-width integers (a bitwise reinterpretation, not a value cast).
    For the GPU wrapper, inserts unrealized_conversion_cast to reinterpret
    memrefs before bufferization.to_tensor (no signature change needed).
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockLowerBlockwiseToPtrPass
    : Pass<"rock-lower-blockwise-to-ptr", "::mlir::func::FuncOp"> {
  let summary = "Lower blockwise load/store ops to pointer-based variants";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockConvertNarrowTypeSignaturesPass
    : Pass<"rock-convert-narrow-type-signatures", "::mlir::func::FuncOp"> {
  let summary = "Convert sub-byte types in function signatures to packed i8";
  let description = [{
    Converts function signatures containing sub-byte memref types
    (e.g. memref<Nxi4>, memref<Nxf4E2M1FN>) to packed wider-type storage
    (e.g. memref<ceil(N/2)xi8>).  Runs before RockEmulateNarrowTypesPass so
    that the block arguments are already converted when the upstream memref
    narrow-type emulation patterns rewrite loads and stores.
  }];
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect",
                           "func::FuncDialect"];
}

def RockEmulateNarrowTypesPass
    : Pass<"rock-emulate-narrow-types", "::mlir::func::FuncOp"> {
  let summary = "Emulate sub-byte types (i4) into packed i8 storage";
  let description = [{
    Uses NarrowTypeEmulationConverter + upstream memref/arith narrow type
    emulation patterns to convert sub-byte memref operations
    (e.g. memref<Nxi4>) into packed wider-type storage (e.g. memref<N/2xi8>)
    with proper bit-manipulation for load/store.

    Must run after RockConvertNarrowTypeSignaturesPass, which converts
    function signatures first.  This ordering avoids a crash in the upstream
    ConvertMemRefLoad/ConvertMemrefStore patterns that call
    extract_strided_metadata on the original (pre-conversion) memref value.
  }];
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "func::FuncDialect",
    "scf::SCFDialect"
  ];
}

def RockMaskNonZeroPreservingFusionsPass
    : Pass<"rock-mask-non-zero-preserving-fusions", "::mlir::func::FuncOp"> {
  let summary = "Insert arith.select masks after non-zero-preserving fusions "
                "fed by Pad/Embed loads";
  let description = [{
    This pass runs AFTER LowerBlockwiseToPtr and BEFORE TransformsToPointerArith.
    When a blockwise load has Pad or Embed transforms, OOB positions are masked
    to zero. If the loaded tile feeds into a non-zero-preserving fusion (e.g.
    arith.addf %tile, 1.0), those positions become non-zero. This pass detects
    such cases and inserts arith.select with the load mask to re-zero OOB
    positions at the end of the fusion chain.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect"];
}

def RockTransformsToPointerArithPass : Pass<"rock-transforms-to-pointer-arith", "::mlir::func::FuncOp"> {
  let summary =
      "Expand transform map chains into pointer arithmetic and validity masks";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect", "triton::TritonDialect"];
}

def RockToTTIRPass : Pass<"rock-to-ttir", "::mlir::func::FuncOp"> {
  let summary = "Convert Rock dialect operations to Triton IR";
  let dependentDialects = [
    "rock::RockDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "triton::TritonDialect"
  ];
}

def RockInsertOutputStoresPass
    : Pass<"rock-insert-output-stores", "::mlir::ModuleOp"> {
  let summary = "Insert rock.store ops for kernel outputs and convert returns "
                "to output arguments";
  let description = [{
    For each kernel function (marked with `rock.kernel`), this pass
    flood-fills forward from each FusionRoot result to identify return
    operands that need stores. For each such return operand:
    1. A new output argument is added to the function.
    2. A rock.store op is inserted before the return that writes to the new
       output argument added in step 1.
    3. The func.return operand is updated to use the rock.store result.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockSerializeHostFuncsPass
    : Pass<"rock-serialize-host-funcs", "::mlir::ModuleOp"> {
  let summary = "Serialize and erase host functions before kernel lowering";
  let description = [{
    Serializes non-kernel (host) functions into a module attribute and erases them
    from the module. Must run before func-level passes that change kernel signatures
    (e.g. RockToTTIRPass sets kernel return to void), so that no func.call to the
    kernel remains when the kernel's type changes.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockFuncToTritonFuncPass : Pass<"rock-func-to-triton-func", "::mlir::ModuleOp"> {
  let summary = "Convert func.func kernels to tt.func with Triton pointer arguments";
  let description = [{
    This pass transforms Rock kernel functions from the standard `func.func` dialect
    to Triton's `tt.func` dialect, preparing them for the Triton compilation pipeline.

    Key transformations:
    1. Converts `func.func` to `tt.func` and `func.return` to `tt.return`
    2. Replaces tensor arguments with `!tt.ptr<element_type>` Triton pointers
    3. Eliminates rock.extract_ptr and adapts the users to use the !tt.ptr directly.
    4. Converts `arith.addi` on pointer tensors to `tt.addptr`
    5. Adds attributes to the tt.func, such as `tt.divisibility` and `tt.pointer_range`
    6. Stores host functions (wrappers, main) as module attributes for later restoration
    7. Preserves grid size as module attributes for gpu.launch_func
  }];
  let dependentDialects = [
    "rock::RockDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "triton::TritonDialect",
  ];
}

def TritonToHsacoPass : Pass<"triton-to-hsaco", "::mlir::ModuleOp"> {
  let summary = "Convert Triton LLVM dialect IR to HSACO binary";
  let description = [{
    This pass translates Triton LLVM dialect IR to HSACO binary format.
    It wraps the TritonToHsaco translation and implements the functionality
    from Triton's compiler.py:
    - make_llir() lines 358-449: LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    - make_amdgcn() lines 452-473: LLVM -> AMDGCN assembly
    - make_hsaco() lines 476-488: AMDGCN assembly -> HSACO binary

    The resulting HSACO binary is stored as a "triton.hsaco" attribute on the module.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
  ];
  let options = [
    Option<"triple", "triple", "std::string", "\"amdgcn-amd-amdhsa\"",
      "Triple">,
    Option<"arch", "arch", "std::string", "\"gfx90a\"",
      "AMDGPU architecture (e.g., gfx90a, gfx942, gfx1100)">,
    Option<"features", "features", "std::string", "\"\"",
      "Arch features">,
    Option<"optLevel", "opt-level", "int", "3",
      "Compiler optimization level">,
    Option<"numWarps", "num-warps", "int", "4",
      "Number of warps per block">,
    Option<"numCTAs", "num-ctas", "int", "1",
      "Number of CTAs">,
    Option<"wavesPerEU", "waves-per-eu", "int", "0",
      "Waves per execution unit hint (0 for default)">,
    Option<"enableFpFusion", "enable-fp-fusion", "bool", "true",
      "Enable FP fusion optimizations">,
    Option<"allowFlushDenorm", "allow-flush-denorm", "bool", "false",
      "Allow flushing denormal numbers">,
    Option<"scalarizePackedFops", "scalarize-packed-fops", "bool", "false",
      "Scalarize packed float operations">,
    Option<"scheduleHint", "schedule-hint", "std::string", "\"none\"",
      "Scheduling hint (e.g., 'memory-bound-attention' for iterative-ilp scheduler)">
  ];
}

def RockRestoreHostCodePass : Pass<"rock-restore-host-code", "::mlir::ModuleOp"> {
  let summary = "Restore host functions after Triton kernel compilation";
  let description = [{
    This pass restores host functions (like the GPU wrapper and main) that were
    stored during RockFuncToTritonFuncPass. It:
    1. Reads the serialized host functions from the module attribute
    2. Parses them back into the module
    3. Updates kernel calls to use the compiled kernel

    This allows the full test harness to work with the Triton-compiled kernel.
  }];
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "func::FuncDialect", "gpu::GPUDialect",
                           "arith::ArithDialect", "memref::MemRefDialect",
                           "ROCDL::ROCDLDialect",
  ];
  let options = [
    Option<"triple", "triple", "std::string", "\"amdgcn-amd-amdhsa\"",
      "Triple">,
    Option<"arch", "arch", "std::string", "\"gfx90a\"",
      "AMDGPU architecture (e.g., gfx90a, gfx942, gfx1100)">,
    Option<"features", "features", "std::string", "\"\"",
      "Arch features">,
    Option<"optLevel", "opt-level", "int", "3",
      "Compiler optimization level">,
  ];
}

def RockInsertOutputFusionLoadsPass : Pass<"rock-insert-output-fusion-loads", "::mlir::func::FuncOp"> {
  let summary = "Insert BlockwiseLoadOp for output fusion loads";
  let description = [{
    This pass runs AFTER ToBlockwise and BEFORE RockLowerLoads.
    It creates rock.load_marker ops used in output fusions.

    Output fusion loads are:
    1. Are NOT reachable from any existing rock.load_marker (input loads)
    2. Feed into fusion ops (arith/math) that operate on the GEMM result

    Example:
      Before (after ToBlockwise):
        %result = scf.for ... {
          %loadedA = rock.blockwise_load %a[indices]
          ...
        }
        %fusionRoot = rock.store_marker %result views [...] [%g, %m, %n]
                        : tensor<64x64xf16> -> tensor<1x128x128xf16>
        %bias_t = rock.transform %bias
        %fused = arith.addf %fusionRoot, %bias_t
        %out = rock.store %fused to %dest

      After:
        %result = scf.for ... { ... }
        %fusionRoot = rock.store_marker %result views [...] [%g, %m, %n]
                        : tensor<64x64xf16> -> tensor<1x128x128xf16>
        %bias_t = rock.transform %bias
        %bias_wrapped = rock.transform %bias_t by <output_grid_subtile>
        %bias_tile = rock.load_marker %bias_wrapped[g, m, n]
        %bias_full = rock.untile %bias_tile : tile -> full
        %fused = arith.addf %fusionRoot, %bias_full
        %out = rock.store %fused to %dest

    The chain block_argument -> transform(s) -> rock.load_marker -> rock.untile allows LowerLoads
    to trace back through block_argument and combine the pre-load transforms with
    the output grid subtile transforms.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect", "math::MathDialect"];
}

def RockRegularizeInputPass
    : Pass<"rock-regularize-input", "::mlir::func::FuncOp"> {
  let summary =
      "Push input fusions past load_markers to simplify load lowering";
  let description = [{
    This pass runs AFTER InsertOutputFusionLoads and BEFORE RockLowerLoads.
    It ensures that each rock.load_marker's source is a pure transform chain
    (no fusion ops), by distributing load_markers past any fusions in their
    source chain.

    For each rock.load_marker whose source traces through fusion ops:
    1. Walk backwards collecting transforms between the marker and the fusions
    2. For each leaf of the fusion DAG (transform chains to block args):
       apply the collected transforms, isolate the chain, create a new
       rock.load_marker
    3. For splat constants: create a tile-shaped constant (no load_marker)
    4. Clone the fusion ops after the load_markers, operating on tile types
    5. Replace the original load_marker with the cloned fusion result

    Example:
      Before:
        %Ta = rock.transform %arg0 by <T0>
        %Tb = rock.transform %arg1 by <T1>
        %fused = arith.addf %Ta, %Tb
        %a = rock.transform %fused by <T2>
        %res = rock.load_marker %a views [<tiling>] [indices]

      After:
        %Ta2 = rock.transform %arg0 by <T0>  // isolated chain
        %Ta2 = rock.transform %Ta by <T2>
        %Tb2 = rock.transform %arg1 by <T1>  // isolated chain
        %Tb2 = rock.transform %Tb by <T2>
        %A0 = rock.load_marker %Ta2 views [<tiling>] [indices]
        %A1 = rock.load_marker %Tb2 views [<tiling>] [indices]
        %res = arith.addf %A0, %A1
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect", "math::MathDialect"];
}

def RockLowerLoadsPass : Pass<"rock-lower-loads", "::mlir::func::FuncOp"> {
  let summary = "Lower rock.load_marker ops to rock.blockwise_load";
  let description = [{
    This pass runs AFTER RockRegularizeInput.
    Each rock.load_marker's source is guaranteed to be a pure transform chain
    (no fusions) by the preceding RegularizeInput pass.

    For each rock.load_marker, the pass applies the marker's extraViews
    transforms on top of the source and creates a rock.blockwise_load.

    Example:
      Before:
        %t = rock.transform %arg0 by <pre_transform>
        %tile = rock.load_marker %t views [<tiling>] [%kIter, %g, %m, %n]

      After:
        %combined = rock.transform %t by <tiling>
        %tile = rock.blockwise_load %combined[%kIter, %g, %m, %n]
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockTosaToElementwisePass
    : Pass<"rock-tosa-to-elementwise", "::mlir::func::FuncOp"> {
  let summary =
      "Convert TOSA elementwise ops directly to arith/math on tensors";
  let description = [{
    This pass converts TOSA pointwise/elementwise operations (e.g. tosa.add,
    tosa.exp, tosa.mul) directly to the corresponding arith and math dialect
    operations on tensor types.  No linalg.generic wrapper is produced.

    The pass only runs on kernel functions (those with the rock.kernel
    attribute).  Non-kernel (CPU) functions are left untouched so the normal
    tosa-to-linalg pipeline can lower them through linalg.generic later.
  }];
  let dependentDialects = ["arith::ArithDialect", "math::MathDialect",
                           "tensor::TensorDialect", "tosa::TosaDialect"];
}

def RockFlattenTosaFuncArgsPass
    : Pass<"rock-flatten-tosa-func-args", "::mlir::func::FuncOp"> {
  let summary = "Flatten N-D tensor function arguments and results to 1-D";
  let description = [{
    Downstream passes require all function boundary tensors to be 1-D.
    The MIGraphX-to-TOSA conversion already produces 1-D boundaries, but
    hand-written TOSA test kernels may have N-D tensor arguments and results.

    This pass rewrites each kernel function so that:
    - Every argument of type tensor<AxBx...xT> becomes tensor<(A*B*...)xT>,
      with a tosa.reshape inserted at the top of the entry block to recover
      the original N-D shape for use inside the function body.
    - Every return operand is flattened with a tosa.reshape to 1-D before the
      func.return, and the function signature is updated accordingly.

    Functions whose boundaries are already 1-D (or scalar) are left unchanged.
  }];
  let dependentDialects = ["tosa::TosaDialect"];
}

def RockLowerStoresPass : Pass<"rock-lower-stores", "::mlir::func::FuncOp"> {
  let summary = "Lower rock.store ops and handle output fusions";
  let description = [{
    This pass processes rock.store ops after GridwiseGemmToBlockwise.
    It handles output fusions by:
    1. Finding rock.store ops and tracing back to blockwise_store
    2. For output fusions (arith/math ops between bst and store):
       - Cloning fusion ops to operate on tiles
       - Creating blockwise_load for other fusion operands
       - Creating new blockwise_store with fused result

    Example (output fusion):
      Before:
        %bst = rock.blockwise_store %tile -> %dest[indices]
        %bias_loaded = rock.load %bias_transformed
        %fused = arith.addf %bst, %bias_loaded
        %out = rock.store %fused to %dest_arg

      After:
        %bias_tile = rock.blockwise_load %bias_transformed[indices]
        %fused_tile = arith.addf %tile, %bias_tile
        %out = rock.blockwise_store %fused_tile -> %dest[indices]
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect", "math::MathDialect"];
}

#endif // MLIR_DIALECT_ROCK_PASSES
