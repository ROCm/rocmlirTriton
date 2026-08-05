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

def RockViewToTransformPass
    : Pass<"rock-view-to-transform", "::mlir::func::FuncOp"> {
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

def RockAffixTuningParametersPass
    : Pass<"rock-affix-params", "::mlir::func::FuncOp"> {
  let summary = "populate tuning parameters for gemm and convolution ops";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockGemmToGridwisePass
    : Pass<"rock-gemm-to-gridwise", "::mlir::func::FuncOp"> {
  let summary = "convert gemm to gridwise gemm, adding padding";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect"];
}

def RockAttnToGridwisePass
    : Pass<"rock-attn-to-gridwise", "::mlir::func::FuncOp"> {
  let summary =
      "This pass converts rock.attention and rock.gemm_elementwise_gemm "
      "into the appropriate rock.gridwise_attention adding padding and "
      "group dimensions if needed.";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect"];
}

def RockFoldBroadcastPass
    : Pass<"rock-fold-broadcast", "::mlir::func::FuncOp"> {
  let summary =
      "Convert Gemm([g, M, K], [broadcast{1}, K, N]) to Gemm([g*M, K], [K, N])";
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

def RockRegularizeInterGemmFusionPass
    : Pass<"rock-regularize-inter-gemm-fusion", "::mlir::func::FuncOp"> {
  let summary = "Regularize elementwise fusion bodies between two GEMMs";
  let description = [{
    This pass regularizes the elementwise body region of gemm-gemm-like ops
    (RockGemmGemmWrapperInterface) so that the body contains only elementwise
    operations and no rock.transform ops.

    Transforms on block arguments are pushed outside the body by applying them
    to the corresponding external inputs. If the first block argument (the
    first GEMM product) has transforms, they are eliminated by applying their
    inverse to all other block arguments' transform chains so that the entire
    body operates in the first GEMM product's original shape.
  }];
  let dependentDialects = ["rock::RockDialect", "arith::ArithDialect"];
}

def RockGridwiseGemmToBlockwisePass
    : Pass<"rock-gridwise-gemm-to-blockwise", "::mlir::func::FuncOp"> {
  let summary = "expand gridwise gemm into blockwise copy, blockwise gemm, and "
                "threadwise copy";
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
  let summary = "This pass modifies fusion ops for split-k fusions. It "
                "converts any arith.addf/arith.subf gemmOut, other to "
                "arith.addf gemmOut, other/splitkFactor";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockAllowFastMathFlagsPass
    : Pass<"rock-allow-fast-math-flags", "::mlir::func::FuncOp"> {
  let summary = "Tag applicable floating-point arith and math ops with "
                "fast-math flags such as arcp, nsz, contract, and afn";
  let dependentDialects = ["arith::ArithDialect", "func::FuncDialect",
                           "math::MathDialect"];
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
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "scf::SCFDialect", "triton::TritonDialect"];
}

def RockLowerBlockwiseToPtrPass
    : Pass<"rock-lower-blockwise-to-ptr", "::mlir::func::FuncOp"> {
  let summary = "Lower blockwise load/store ops to pointer-based variants";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockDecomposeNonPow2TilesPass
    : Pass<"rock-decompose-nonpow2-tiles", "::mlir::func::FuncOp"> {
  let summary = "Split non-power-of-two gridwise GEMM tiles into power-of-two "
                "sub-gridwise_gemms (M/N peeling)";
  let description = [{
    Rock tuning params may legally carry non-power-of-two tiles (e.g. an
    mPerBlock/nPerBlock of 80), but the Triton layouts later produced by
    RockToTTIR require power-of-two shapes. This pass splits every
    rock.gridwise_gemm whose mPerBlock or nPerBlock is not a power of two into a
    grid of power-of-two-tile sub-gridwise_gemms.

    Runs after GemmToGridwise/AttnToGridwise (so it sees canonical
    rock.gridwise_gemm ops with tuning params and the func grid_size) and before
    GridwiseGemmToBlockwise (so the IR is still full-tensor gridwise ops, with no
    loops, blockwise ops, or Triton ops to reason about).

    1. mPerBlock/nPerBlock are decomposed into minimal power-of-two segments
       (80 -> {64, 16}), giving an (m-segment, n-segment) grid of cells.
    2. Per cell, A (G x M x K) is sliced to its M-segment and B (G x K x N) to
       its N-segment (reusing each A-segment across a row of cells and each
       B-segment down a column), and a sub-gridwise_gemm with a pow2 tile is
       emitted. The output fusion chain and stores are replicated per cell, with
       extra fusion inputs sliced to match. Input fusion needs no handling: it
       rides on the sliced A/B views and is tiled (pow2) downstream.

    Every sub-op keeps the same mBlocks/nBlocks/grid_size, so the grid layout is
    unchanged. The segments exactly cover the original extent (64 + 16 = 80), so
    no padding is introduced. Each sub-op lowers to its own K-loop downstream;
    merging those sibling loops is a separate follow-up.
  }];
  let dependentDialects = ["rock::RockDialect", "arith::ArithDialect"];
}

def RockFuseSiblingLoopsPass
    : Pass<"rock-fuse-sibling-loops", "::mlir::func::FuncOp"> {
  let summary =
      "Fuse independent sibling scf.for K-loops with identical bounds";
  let description = [{
    DecomposeNonPow2Tiles splits one non-power-of-two rock.gridwise_gemm into
    several power-of-two sub-gemms, each of which GridwiseGemmToBlockwise lowers
    to its own scf.for K-loop with identical bounds (K/kPerBlock, step 1). These
    sibling loops reload the shared operand (B for an M-split, A for an N-split).

    This pass fuses sibling scf.for loops that live in the same block and have
    numerically identical lower/upper bounds and step into a single loop, as
    long as they are mutually independent (neither loop transitively uses the
    other's results). Bounds are compared by constant value rather than SSA
    identity, because each sub-gemm builds its own bound constants. Running CSE
    afterwards deduplicates the now-co-located shared operand loads.
  }];
  let dependentDialects = ["rock::RockDialect", "scf::SCFDialect"];
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
  let dependentDialects = ["affine::AffineDialect", "arith::ArithDialect",
                           "memref::MemRefDialect", "func::FuncDialect",
                           "scf::SCFDialect"];
}

def RockPreserveMaskedLoadSemanticsPass
    : Pass<"rock-preserve-masked-load-semantics", "::mlir::func::FuncOp"> {
  let summary = "Preserve masked-load OOB semantics across fusion chains";
  let description = [{
    This pass runs AFTER LowerBlockwiseToPtr and BEFORE TransformsToPointerArith.
    When a blockwise load has Pad or Embed transforms, OOB positions are masked
    to zero. If the loaded tile feeds into a non-zero-preserving fusion (e.g.
    arith.addf %tile, 1.0), those positions become non-zero. Zero-preserving
    fusions can also need remasking when they feed `rock.blockwise_reduce max`,
    because zero is not max-neutral. This pass detects such cases and inserts
    arith.select with the load mask at non-fusion uses of the fusion chain. The
    masked-out value is chosen for the consumer: zero by default, or the type's
    smallest representable value for `rock.blockwise_reduce max` (e.g. -inf for
    IEEE floats, the most negative finite value for FP8 formats without
    infinity, signed INT_MIN for signed integers) so masked-out lanes do not
    influence the reduction.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect"];
}

def RockCollapseContiguousMergesPass
    : Pass<"rock-collapse-contiguous-merges", "::mlir::func::FuncOp"> {
  let summary = "Collapse contiguous merges in rock.transform chains";
  let description = [{
    This pass runs BEFORE TransformsToPointerArith.

    When a `Merge` splits a dimension whose lower dimensions are actually
    contiguous in memory, it fuses them back into a single wider dimension
    (see `collapseContiguousMerges`).

    Why this matters: each element's memory offset is computed from the
    dimension index. We say the dimension is "stride-1" when stepping the index
    by 1 moves the offset by exactly 1, i.e. neighboring elements sit at
    neighboring addresses. Triton looks for this stride-1 pattern to prove the
    access is contiguous and read many elements in one wide load instead of many
    small ones.

    If the contiguous dimension is split apart and rebuilt (as in an im2col-style
    transform), the offset is no longer stride-1: the step is +1 within each
    piece but jumps when one piece wraps into the next. Triton can no longer
    recognize the contiguity and falls back to small per-element loads. Keeping
    the dimension intact preserves the stride-1 offset and the wide loads that
    the split form prevents.

    Example: a buffer viewed as `Unmerge{2, 3, 8}` ["block", "a", "b"] with a
    `Merge{3, 8}` ["tile"] -> ["a", "b"] tile dimension. Here "a" (stride 8) and
    "b" (stride 1, size 8) are adjacent in memory, so the {a, b} tile is one
    contiguous run of 24 elements. The pass collapses it to `Merge{1, 24}` and
    widens the view to `Unmerge{2, 1, 24}`, exposing the stride-1 run of 24.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockTransformsInvariantCodeMotionPass
    : Pass<"rock-transforms-invariant-code-motion", "::mlir::func::FuncOp"> {
  let summary =
      "Hoist loop-invariant transforms_to_ptr arithmetic out of scf.for loops";
  let description = [{
    Loop-invariant code motion for `rock.transforms_to_ptr` ops nested 
    inside `scf.for` loops. Runs before TransformsToPointerArith.

    The pass hoists only if 2 conditions are met:
    1. The pointer offset is an affine-linear function of the loop induction variable.
    2. The validity mask is loop-invariant

    The rewrite replaces in loop pointer recomputation with a loop-invariant base + 
    one constant stride per iteration, and computes the mask once before the loop. 
    That's only valid when:
    - The per-iteration offset change is one constant: this breaks for a non-contiguous
      Merge because the increment is not constant, they vary depending on specific values
      of the loop counter, not on every step, making the real increment position-dependent.
    - The mask is identical every iteration: which breaks when a Pad/Embed decides
      validity by checking a coordinate that moves as the IV increases: the set of valid
      elements is then different on each iteration, so the mask would have to be recomputed
      every step and cannot be precomputed once.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "scf::SCFDialect", "arith::ArithDialect",
                           "triton::TritonDialect"];
}

def RockTransformsToPointerArithPass
    : Pass<"rock-transforms-to-pointer-arith", "::mlir::func::FuncOp"> {
  let summary =
      "Expand transform map chains into pointer arithmetic and validity masks";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "triton::TritonDialect"];
}

def RockToTTIRPass : Pass<"rock-to-ttir", "::mlir::func::FuncOp"> {
  let summary = "Convert Rock dialect operations to Triton IR";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect", "triton::TritonDialect"];
}

def RockSortDimensionsMemoryLayoutPass
    : Pass<"rock-sort-dimensions-memory-layout", "::mlir::func::FuncOp"> {
  let summary = "Sort dimensions by stride of the underlying memory layout";
  let description = [{
    Reorders tensor dimensions to match physical memory layout (descending stride
    order) for GEMM, Conv, Attention, and their fused variants.

    For example, if a GEMM's B matrix has physical memory layout [G, N, K] but
    the logical view is [G, K, N], the pass inserts a transform to make the
    logical order match physical memory, then marks B as transposed in the
    rock.gemm op.

    The pass traces through rock.transform chains and shape-preserving
    elementwise ops to determine actual memory strides, then inserts transforms
    to align logical dimension order with physical memory. Transpose flags (for
    GEMM-like ops) or layout attributes (for Conv) are updated accordingly.

    This normalization ensures downstream passes (especially vectorization)
    correctly analyze memory access patterns.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect"];
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

def RockTensorToTritonPtrPass
    : Pass<"rock-tensor-to-triton-ptr", "::mlir::ModuleOp"> {
  let summary = "Convert tensor semantic kernels (rock) to pointer semantic "
                "kernels (triton)";
  let description = [{
    This pass transforms kernels from tensor semantic to pointer semantic.

    Key transformations:
    1. Converts `arith.addi` on pointer tensors to `tt.addptr`
    2. Replaces tensor arguments with `!tt.ptr<element_type>` Triton pointers
    3. Eliminates rock.extract_ptr and adapts the users to use the !tt.ptr directly.
    4. Adds attributes to the tt.func, such as `tt.divisibility` and `tt.pointer_range`
    5. Converts `func.func` to `tt.func` and `func.return` to `tt.return`
    6. Stores host functions (wrappers, main) as module attributes for later restoration
    7. Preserves grid size as module attributes for gpu.launch_func
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect", "triton::TritonDialect",
  ];
}

def ResolveKernelLaunchParamsPass
    : Pass<"resolve-kernel-launch-params", "::mlir::ModuleOp"> {
  let summary =
      "Convert dynamic LDS to static and remove unused workspace args";
  let description = [{
    Triton's AMDGPU lowering creates two runtime artifacts that can be resolved
    statically at compile time:

    1. **Dynamic shared memory (LDS)** — `@global_smem` is declared as an
       `external addrspace(3) global [0 x i8]`, requiring the host to pass the
       LDS size at kernel launch.  This pass rewrites it to
       `internal addrspace(3) global [N x i8] undef` (where N comes from the
       `ttg.shared` module attribute) so the AMDGPU backend bakes the size into
       `.amdhsa_group_segment_fixed_size` in the kernel descriptor.

    2. **Unused workspace arguments** — Triton appends global-scratch and
       profile-scratch pointer arguments to every kernel.  If they are unused
       in the IR, this pass strips them from the function signature.

    The pass also validates that `ttg.shared` does not exceed the hardware LDS
    limit for the target architecture (read from the `rock.arch` attribute on
    the kernel function).

    After this pass the `ttg.shared` attribute is removed (shared memory is now
    static), so downstream launch code should pass 0 for dynamic shared memory.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
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
  let dependentDialects = ["LLVM::LLVMDialect", ];
  let options =
      [Option<"triple", "triple", "std::string", "\"amdgcn-amd-amdhsa\"",
              "Triple">,
       Option<"arch", "arch", "std::string", "\"gfx90a\"",
              "AMDGPU architecture (e.g., gfx90a, gfx942, gfx1100)">,
       Option<"features", "features", "std::string", "\"\"", "Arch features">,
       Option<"optLevel", "opt-level", "int", "3",
              "Compiler optimization level">,
       Option<"numWarps", "num-warps", "int", "4", "Number of warps per block">,
       Option<"numCTAs", "num-ctas", "int", "1", "Number of CTAs">,
       Option<"wavesPerEU", "waves-per-eu", "int", "0",
              "Waves per execution unit hint (0 for default)">,
       Option<"enableFpFusion", "enable-fp-fusion", "bool", "true",
              "Enable FP fusion optimizations">,
       Option<"allowFlushDenorm", "allow-flush-denorm", "bool", "true",
              "Allow flushing denormal numbers">,
       Option<"scalarizePackedFops", "scalarize-packed-fops", "bool", "false",
              "Scalarize packed float operations">,
       Option<"llvmFnAttrs", "llvm-fn-attrs", "std::string", "\"\"",
              "Debug-only: comma-separated LLVM function attributes (name or "
              "name=value) applied to the kernel after Triton's attributes. "
              "Mirrors the experimental llvm_fn_attrs HIPOption in "
              "compiler.py.">,
       Option<"useExpertScheduling", "use-expert-scheduling", "int", "-1",
              "Tri-state override of the AMDGPU expert-scheduling codegen flag "
              "(-1=arch default (on for gfx1250), 0=off, 1=on).">];
}

def RockEmitGpuBinaryPass : Pass<"rock-emit-gpu-binary", "::mlir::ModuleOp"> {
  let summary =
      "Emit GPU binary and wire kernel launches after Triton compilation";
  let description = [{
    After Triton lowers kernels to LLVM IR and `TritonToHsacoPass` produces
    HSACO, this pass materializes a `gpu.binary` from that HSACO (with kernel
    metadata), marks the module as a GPU container, and when host functions
    were serialized by `RockTensorToTritonPtrPass`, restores them from
    `rock.host_functions` and rewrites `func.call` to `gpu.launch_func`.

    The `gpu.binary` is emitted whenever kernels and HSACO are present, even if
    no host functions are restored, so downstream consumers can read kernel
    metadata from the module.
  }];
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "func::FuncDialect", "gpu::GPUDialect",
                           "arith::ArithDialect", "memref::MemRefDialect",
                           "ROCDL::ROCDLDialect",
  ];
  let options = [Option<"triple", "triple", "std::string",
                        "\"amdgcn-amd-amdhsa\"", "Triple">,
                 Option<"arch", "arch", "std::string", "\"gfx90a\"",
                        "AMDGPU architecture (e.g., gfx90a, gfx942, gfx1100)">,
                 Option<"features", "features", "std::string", "\"\"",
                        "Arch features">,
                 Option<"optLevel", "opt-level", "int", "3",
                        "Compiler optimization level">,
  ];
}

def RockInsertOutputFusionLoadsPass
    : Pass<"rock-insert-output-fusion-loads", "::mlir::func::FuncOp"> {
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
    : Pass<"rock-flatten-tosa-func-args", "::mlir::ModuleOp"> {
  let summary = "Flatten N-D tensor function arguments and results to 1-D";
  let description = [{
    Downstream passes require all function boundary tensors to be 1-D.
    The MIGraphX-to-TOSA conversion already produces 1-D boundaries, but
    hand-written TOSA test functions may have N-D tensor arguments and results.

    This pass rewrites each function so that:
    - Every argument of type tensor<AxBx...xT> becomes tensor<(A*B*...)xT>,
      with a tosa.reshape inserted at the top of the entry block to recover
      the original N-D shape for use inside the function body.
    - Every return operand is flattened with a tosa.reshape to 1-D before the
      func.return, and the function signature is updated accordingly.

    Functions whose boundaries are already 1-D (or scalar) are left unchanged.
    Functions that have symbol uses within the module are skipped to avoid
    breaking call sites.
  }];
  let dependentDialects = ["func::FuncDialect", "tosa::TosaDialect"];
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

def RockAddTritonMetadataPass
    : Pass<"rock-add-triton-metadata", "::mlir::func::FuncOp"> {
  let summary = "Attach Triton-bound metadata to rock ops before RockToTTIR";
  let description = [{
    Central place to attach metadata that needs to be computed at the rock
    level but consumed later in the Triton pipeline. The pass runs after
    `rock-lower-stores`, while rock ops still carry the structural information
    (transformed views, store destinations, etc.) that gets lost once the IR is
    lowered to TTIR. Each piece of metadata is recorded as a discardable
    attribute so that `RockToTTIR` can forward it onto the corresponding Triton
    op and downstream passes can query it.

    Metadata currently produced:

    - `rock.o_transposed` (output orientation): for every
      `rock.blockwise_gemm` consumed by a store, the pass inspects the store
      destination's vectorization (via `getMaxVectorization` on the fast M and
      N output dimensions) to determine whether the output is row-major (N
      fast) or "transposed"/column-major (M fast), and records the result on
      the `rock.blockwise_gemm`. `RockToTTIR` copies it onto the resulting
      `tt.dot` / `tt.dot_scaled`, and `rock-set-matmul-output-transpose` later
      uses it to select the accelerator result layout so the epilogue store
      coalesces. GEMMs with no reachable store (e.g. a chained-dot head) are
      left untouched so the downstream default layout is kept.

    Additional kinds of Triton-bound metadata can be added here as the pipeline
    grows.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockSetMatmulOutputTransposePass
    : Pass<"rock-set-matmul-output-transpose", "::mlir::ModuleOp"> {
  let summary = "Select the matmul accelerator result layout from "
                "rock.o_transposed";
  let description = [{
    Runs immediately after Triton's `tritonamdgpu-accelerate-matmul`, which
    rewrites `tt.dot` into an accelerator-encoded dot (e.g. WMMA) using a fixed
    default result transposition.

    For each accelerator dot that carries the `rock.o_transposed` metadata (set
    by `rock-add-triton-metadata` and preserved across the accelerate-matmul
    rewrite), this pass overrides the result layout's transposition so the
    consuming epilogue store can be vectorized/coalesced.  The orientation is
    chosen from the `rock.o_transposed` attribute computed at the rock level,
    instead of walking forward to the store.

    Currently only the WMMA (`ttg.amd_wmma`) result layout is adjusted; other
    accelerator layouts keep their default. The version-dependent mapping is:
      row-major output (oTransposed = false): isTransposed = (version > 1)
      col-major output (oTransposed = true):  isTransposed = (version == 1)
  }];
  let dependentDialects = ["rock::RockDialect", "triton::TritonDialect",
                           "triton::gpu::TritonGPUDialect"];
}

def RockSetReductionLayoutPass
    : Pass<"rock-set-reduction-layout", "::mlir::ModuleOp"> {
  let summary = "Redistribute the reduction-operand load layout onto the "
                "reduction dim";
  let description = [{
    Redistribute the reduction-operand global-load `#blocked`
    layout so that each warp owns a full reduction (K) strip. Moving
    the load's `warpsPerCTA` off the free/N dim and onto the reduction/K dim
    can potentially reduce SGPR spill on convolution kernels.

    This only applies to `tt.load`/`amdgpu.buffer_load` staged through an
    `amdgpu.in_thread_transpose`, whose re-layout decouples the read
    distribution from the LDS layout the dot consumes; direct-to-LDS and TDM
    loads have no such stage and are out of scope.
  }];
  let dependentDialects = ["rock::RockDialect", "triton::gpu::TritonGPUDialect",
                           "triton::amdgpu::TritonAMDGPUDialect",
                           "arith::ArithDialect"];
  let options = [Option<
      "useReductionLayout", "use-reduction-layout", "int64_t", /*default=*/"-1",
      "Tri-state gate: -1 (default / heuristic) rewrites only convolution "
      "kernels (those carrying the `rock.conv_kernel` attribute); 0 disables "
      "the rewrite entirely (no kernel is touched); 1 forces the rewrite on "
      "every kernel.">];
}

def RockAnalyzeMemoryUsePass
    : Pass<"rock-analyze-memory-use", "::mlir::func::FuncOp"> {
  let summary = "Analyze kernel argument memory usage and set attributes";
  let description = [{
    Walks kernel function arguments (those of `RankedTensorType`) to determine
    their memory access patterns and attaches LLVM attributes plus Triton
    metadata accordingly. Runs on functions marked with `rock.kernel`; other
    functions are left untouched.

    Memory-access classification (mutually exclusive):

    - `llvm.readonly`: argument is only loaded from
      (no `rock.blockwise_store` writes to it).
    - `llvm.writeonly`: argument is only stored to via `StoreMethod::Set`
      (pure overwrite, never read).
    - `llvm.readnone`: argument is never accessed.
    - *(none of the above)*: argument is read and written via atomic store
      (`atomic_add`/`atomic_max`); the implicit read-modify-write makes
      classifying it as readonly/writeonly unsound.

    Always-on attributes (set on every tensor argument):

    - `llvm.noalias`, `llvm.nocapture`, `llvm.nofree`, `llvm.nonnull`,
      `llvm.noundef`
    - `llvm.align = 16` (kernel pointers are assumed 16-byte aligned)
    - `tt.divisibility = 16` (Triton vectorization hint matching the
      alignment)

    Static-shape-only attributes:

    - `llvm.dereferenceable = ceil(numElements * bitWidth / 8)`
    - `tt.pointer_range = 32` when the tensor fits in a 32-bit offset range
      (< 2 GiB), enabling Triton's buffer-ops conversion.

    Note: `llvm.inreg` (for SGPR argument preloading) is *not* set here; it
    is added later by `TritonToHsaco` for both tensor and scalar arguments.

    Also emits an error if an argument is both read and written in a
    non-atomic context, which violates our kernel argument aliasing
    assumptions.
  }];
  let dependentDialects = ["rock::RockDialect", "LLVM::LLVMDialect"];
}

def RockPrepareLLVMPass
    : Pass<"rock-prepare-llvm", "::mlir::LLVM::LLVMFuncOp"> {
  let summary = "Annotate LLVM IR for efficient AMDGPU codegen";
  let description = [{
    This pass annotates LLVM dialect IR produced by the Triton lowering
    pipeline with metadata that enables more efficient code generation
    by the AMDGPU backend:

    1. Marks GEP operations as `inbounds` (excluding buffer fat pointers).
    2. Marks loads from readonly arguments as invariant and attaches alias
       scope metadata to work around an AMDGPU backend issue where noalias
       attributes on kernel arguments are discarded.
    3. Annotates atomic RMW operations with `no_fine_grained_memory`,
       `no_remote_memory`, and `ignore_denormal_mode` metadata to enable
       native hardware atomic instructions instead of CAS fallbacks.
    4. Marks every kernel `amdgpu-no-heap-ptr`. Rock kernels never call a
       device allocator (`malloc`/`free`/`__ockl_dm_*`), so they can never
       reach the rocclr device heap; dropping the hidden heap implicit arg
       stops the runtime from launching the one-time `__amd_rocclr_initHeap`
       setup kernel at module load.
  }];
  let options = [Option<"allowFlushDenorm", "allow-flush-denorm", "bool",
                        /*default=*/"true",
                        "Set ignore_denormal_mode on atomic RMW ops">];
  let dependentDialects = ["LLVM::LLVMDialect", "ROCDL::ROCDLDialect"];
}

#endif // MLIR_DIALECT_ROCK_PASSES
