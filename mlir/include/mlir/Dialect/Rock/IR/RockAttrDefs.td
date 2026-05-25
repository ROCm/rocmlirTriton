//===- RockAttrDefs.td - Rock MLIR Attributes ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_ATTRS
#define ROCK_ATTRS

include "mlir/Dialect/Rock/IR/RockBase.td"
include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//TODO: submit this to upstream
class ArrayRefOfStringRefParameter<string desc = ""> :
    AttrOrTypeParameter<"::llvm::ArrayRef<::llvm::StringRef>", desc> {
  let allocator = [{ {
    ::llvm::SmallVector<::llvm::StringRef> tmpFields;
    size_t len = $_self.size();
    tmpFields.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      tmpFields.push_back($_allocator.copyInto($_self[i]));
    }
    $_dst = $_allocator.copyInto(::llvm::ArrayRef<StringRef>(tmpFields));
  } }];
  let cppStorageType = "::llvm::SmallVector<StringRef>";
}

def IndexArrayAttr : TypedArrayAttrBase<IndexAttr,
                                      "Index integer array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

class Rock_I32Enum<string name, string summary, list<I32EnumAttrCase> cases>
  : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::rock";

  let genSpecializedAttr = 0;
  let stringToSymbolFnName = "get" # name # "ForName";
  let symbolToStringFnName = "getNameFor" # name;
}

class Rock_Attr<string name, list<Trait> traits = []> :
    AttrDef<Rock_Dialect, name, traits>;

/// ConvOpType
def ConvOpType : I32EnumAttrCase<"Fwd", 0, "conv">;
def ConvOpBwdDataType : I32EnumAttrCase<"BwdData", 1, "conv_bwd_data">;
def ConvOpBwdWeightType : I32EnumAttrCase<"BwdWeight", 2, "conv_bwd_weight">;

def ConvOpTypes : Rock_I32Enum<"ConvOpType", "The type of a convolution operation",
  [ConvOpType, ConvOpBwdDataType, ConvOpBwdWeightType]>;

/// Kerneltype
def KernelTypeConv : I32EnumAttrCase<"Conv", 0>;
def KernelTypeConvBwdData : I32EnumAttrCase<"ConvBwdData", 1>;
def KernelTypeConvBwdWeight : I32EnumAttrCase<"ConvBwdWeight", 2>;
def KernelTypeGemm : I32EnumAttrCase<"Gemm", 3>;
def KernelTypeAttention : I32EnumAttrCase<"Attention", 4>;
def KernelTypeGemmElementwiseGemm : I32EnumAttrCase<"GemmElementwiseGemm", 5>;
def KernelTypeConvElementwiseGemm : I32EnumAttrCase<"ConvElementwiseGemm", 6>;

def KernelType
    : Rock_I32Enum<"KernelType", "Any of the possible types of a rock kernel",
                   [KernelTypeConv, KernelTypeConvBwdData,
                    KernelTypeConvBwdWeight, KernelTypeGemm,
                    KernelTypeAttention, KernelTypeGemmElementwiseGemm,
                    KernelTypeConvElementwiseGemm]>;

/// TransformType
def PassThrough : I32EnumAttrCase<"PassThrough", 0>;
def Pad : I32EnumAttrCase<"Pad", 1>;
def Slice : I32EnumAttrCase<"Slice", 2>;
def Embed : I32EnumAttrCase<"Embed", 3>;
def Unmerge : I32EnumAttrCase<"Unmerge", 4>;
def Merge : I32EnumAttrCase<"Merge", 5>;
def AddDim : I32EnumAttrCase<"AddDim", 7>;
def Broadcast : I32EnumAttrCase<"Broadcast", 8>;
def ConstDim : I32EnumAttrCase<"ConstDim", 9>;

def TransformType : Rock_I32Enum<"TransformType",
    "The operation type for a coordinate transformation",
    [PassThrough, Pad, Slice, Embed, Unmerge, Merge, AddDim, Broadcast,
     ConstDim]>;

/// StoreMethod

def StoreMethod_Set : I32EnumAttrCase<"Set", 0, "set">;
def StoreMethod_AtomicAdd : I32EnumAttrCase<"AtomicAdd", 1, "atomic_add">;
def StoreMethod_AtomicMax : I32EnumAttrCase<"AtomicMax", 2, "atomic_max">;

def StoreMethod : Rock_I32Enum<"StoreMethod",
    "What operation should be used to write data to memory",
    [StoreMethod_Set, StoreMethod_AtomicAdd, StoreMethod_AtomicMax]>;

def StoreMethodAttr : EnumAttr<Rock_Dialect, StoreMethod, "StoreMethod">;

/// ReduceMethod

def ReduceMethod_Sum : I32EnumAttrCase<"Sum", 0, "sum">;
def ReduceMethod_Max : I32EnumAttrCase<"Max", 1, "max">;

def ReduceMethod : Rock_I32Enum<"ReduceMethod",
    "What operation should be used for the reduction",
    [ReduceMethod_Sum, ReduceMethod_Max]>;

def ReduceMethodAttr : EnumAttr<Rock_Dialect, ReduceMethod, "ReduceMethod">;

/// TransformAttr
def Rock_TransformAttr : Rock_Attr<"Transform"> {
    let mnemonic = "transform";

    let description = [{
        A coordinate transformation, that is, a component of a transform_map.

        A transform_map is a way to describe affine maps between coordinate spaces
        that easily allows one to "take their derivative" when emitting hot unrolled
         loops and that simplifies the process of describing these mappings.

        Each transform produces some number of upper dimensions from some
        number of lower dimensions. A collection of transformation that covers a set
        of upper and lower dimensions can be packaged into a transform_map,
        at which point they specify an affine map between coordinate spaces.

        Each transform is specified as an attribute with the form
        ```
          #rock.transform<Type{param1, param2} ["upper1", "upper2"] at [dim1, dim2] -> ["lower1", "lower2"] at [dim3, dim4]>
        ```
        where the params block may be omitted for transforms that don't take parameters.

        The available transform types are
        - PassThrough, which maps N lower dimensions to N upper dimensions
            without changing the coordinates in those dimensions
        - Pad{l1, r1, l2, r2, ..., lN, rN} Adds padding to N dimensions in the
          transformed upper output. The parameters are the amount of padding to
          add to the left and right of each dimension, concatenated
        - Slice{b1, e1, b2, e2, ... bN, eN} Takes a slice of N dimensions
          creating, for each upper dimension i, a dimension of size e_i - b_i where 0
          in the upper layer maps to b_i in the lower layer
        - Embed{c1, c2, ... cN} - Creates N upper dimensions from 1 lower dimension
          with the map (d1, d2, ..., dN) -> (d1 * C1 + d2 * C2 + ... dN * cN)
        - Unmerge{c1, c2, ..., cN} - Creates N upper dimensions as in Embed,
          but uses the map (d1, d2, ..., dN) -> (dN + cN * (d_{N - 1} * (... d1)))
        - Merge{c1, c2, ..., cN} - Combines N lower dimensions into 1 upper dimension
          with the parameters holding the sizes of the lower dimensions.
          This has the effect of concatentating the dimensions if they are consecutive
        - AddDim{size} - Adds one upper dimension of the given size that does not
          correspond to any lower dimension.
        - Broadcast{c1, c2, ..., cN} - Broadcast lower dimensions of size
          c1, c2, ... cN to upper dimensions whose size isn't (currently) recorded
          in the parameters. Note that the upper dimension's length doesn't have to
          evenly divide into the lower one.
        - ConstDim{c1, l1, c2, l2, .. cN, lN} an operator that sets the lower
          dimensions of length l1, ... lN to the values c1, ..., cN. It takes no
          input arguments.
    }];

    let parameters = (ins
    AttrParameter<"::mlir::rock::TransformType",
        "The type of transformation to perform">:$type,
    ArrayRefParameter<"int64_t",
        "The parameters to the transformation, if any">:$params,
    ArrayRefOfStringRefParameter<
        "The names of the upper dimensions (input arguments) of this transform">:$upperNames,
    ArrayRefParameter<"uint32_t",
        "The indices of the upper dimensions (input arguments) of this transform">:$upperDims,
    ArrayRefOfStringRefParameter<
        "The names of the lower dimensions (outputs) of this transformation">:$lowerNames,
    ArrayRefParameter<"uint32_t",
        "The indices of the lower dimensions (outputs) of this transformation">:$lowerDims
    );

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

/// TransformsAttr
def Rock_TransformMapAttr : Rock_Attr<"TransformMap"> {
    let mnemonic = "transform_map";
    let description = [{
        A package of a series of coordinate transformation descriptions
        that completely defines how to map from some upper dimensions P
        to some other lower dimensions Q.

        Comes packaged with an affine map describing the transformation and upper and lowr bounds;
    }];

    let parameters = (ins
    ArrayRefParameter<"::mlir::rock::TransformAttr",
        "The individual coordinate transformations">:$ops,
    AttrParameter<"AffineMapAttr", "The composed map describing the `ops`">:$map,
    // Thes aren't strictly needed and can probably be removed if we're willing to
    // pay an ergonomics penalty.
    DefaultValuedParameter<"DenseI64ArrayAttr", "$_builder.getDenseI64ArrayAttr({})">:$upperBounds,
    DefaultValuedParameter<"DenseI64ArrayAttr", "$_builder.getDenseI64ArrayAttr({})">:$lowerBounds
    );

    let builders = [
      AttrBuilderWithInferredContext<
        (ins "ArrayRef<::mlir::rock::TransformAttr>":$transforms,
            "ArrayRef<int64_t>":$upperBounds,
            "ArrayRef<int64_t>":$lowerBounds)>,
    ];

    let assemblyFormat = [{
        `<` $map `by` ` ` `[` $ops `]` `bounds` `=` ($upperBounds^) : (`[` `]`)? `->` ($lowerBounds^) : (`[` `]`)? `>`
    }];
    let genVerifyDecl = 1;
}

def Rock_GemmParamsAttr
    : Rock_Attr<"GemmParams", [RockTuningParamAttrInterface]> {
  let mnemonic = "gemm_params";
  let description = [{
    The tuning parameters for matrix multiplication.

    - mPerBlock: The number of values of m to process in each workgroup
    - nPerBlock: The number of values of n to process in each workgroup
    - kPerBlock: The number of values of k to process in each workgroup
    - kpack: The number of matrix instructions we can run with one LDS load (can be 1 or 2, see AccelerateAMDMatmul.cpp)
    - numCTAs: The number CTAs
    - numWaves: The number waves
    - matrixInstrNonkdim: The size of the m/n dimension in the accelerator (XDL)
      instruction
    - splitKFactor: Split-k factor for the Split-k GEMM algorithm
    - numStages: Number of stages
    - wavesPerEU: Hint to backend compiler for wavefronts per execution unit
    - gridGroupSize: Number of chiplets to group together to perform a spatially
      local tile
    - useAsyncCopy: Overrides the per-arch default application for the async-copy schedule.
    - useBlockPingpong: Overrides the per-arch default application for the block-pingpong schedule.
    - useInThreadTranspose: Overrides the per-arch default application for the in-thread-transpose pass.
    - useBufferOps: Overrides the convert-to-buffer-ops / canonicalize-pointers / optimize-buffer-op-ptr
      pass cluster.
    - useBufferAtomics: Overrides buffer atomics in convert-to-buffer-ops (requires useBufferOps
      to be enabled).
    - bufferOpsAnalyzeSmallTensorRange: Overrides small-tensor range analysis in
      convert-to-buffer-ops (requires useBufferOps to be enabled).
    - scheduleHint: Bitfield selector mirroring upstream Triton's
      `HIPOptions.schedule_hint`, which is a comma-separated multi-select
      string ("attention,memory-bound-attention"). Encoded as a bitfield so
      that combinations are expressible end-to-end.
      kKnobDefault=-1=arch default, 0=none, bit 0 (=1)=attention,
      bit 1 (=2)=memory-bound-attention. Bit assignments are append-only;
      new variants must claim a new bit and the table in
      KnobUtils.cpp must be extended in lockstep.

    NOTE: The following fields are serialized into the perfConfig but they are
    not actually tuned. Instead, they are always set to `0` (use the backend heuristic)
    or `-1` (the knob default) by the tuner:

    - `wavesPerEU`                       (set to 0)
    - `gridGroupSize`                    (set to 0)
    - `useAsyncCopy`                     (set to -1)
    - `useBlockPingpong`                 (set to -1)
    - `useInThreadTranspose`             (set to -1)
    - `useBufferOps`                     (set to -1)
    - `useBufferAtomics`                 (set to -1)
    - `bufferOpsAnalyzeSmallTensorRange` (set to -1)
  }];

  let parameters = (ins "int64_t":$mPerBlock, "int64_t":$nPerBlock,
      "int64_t":$kPerBlock, "int64_t":$kpack, "int64_t":$numCTAs,
      "int64_t":$numWaves, "int64_t":$matrixInstrNonkdim,
      "int64_t":$splitKFactor, "int64_t":$numStages, "int64_t":$wavesPerEU,
      "int64_t":$gridGroupSize,
      DefaultValuedParameter<"int64_t", "-1">:$useAsyncCopy,
      DefaultValuedParameter<"int64_t", "-1">:$useBlockPingpong,
      DefaultValuedParameter<"int64_t", "-1">:$useInThreadTranspose,
      DefaultValuedParameter<"int64_t", "-1">:$useBufferOps,
      DefaultValuedParameter<"int64_t", "-1">:$useBufferAtomics,
      DefaultValuedParameter<"int64_t", "-1">:$bufferOpsAnalyzeSmallTensorRange,
      DefaultValuedParameter<"int64_t", "-1">:$scheduleHint);

  let extraClassDeclaration = [{
    void getPerfConfigStr(SmallVectorImpl<char> &perfStr) {
        ("gemm:v2:"
        + Twine(getMPerBlock()) + ","
        + Twine(getNPerBlock()) + ","
        + Twine(getKPerBlock()) + ","
        + Twine(getKpack()) + ","
        + Twine(getNumCTAs()) + ","
        + Twine(getNumWaves()) + ","
        + Twine(getMatrixInstrNonkdim()) + ","
        + Twine(getSplitKFactor()) + ","
        + Twine(getNumStages()) + ","
        + Twine(getWavesPerEU()) + ","
        + Twine(getGridGroupSize()) + ","
        + Twine(getUseAsyncCopy()) + ","
        + Twine(getUseBlockPingpong()) + ","
        + Twine(getUseInThreadTranspose()) + ","
        + Twine(getUseBufferOps()) + ","
        + Twine(getUseBufferAtomics()) + ","
        + Twine(getBufferOpsAnalyzeSmallTensorRange()) + ","
        + Twine(getScheduleHint()))
        .toVector(perfStr);
    }

    StringAttr getPerfConfigAttr() {
        SmallVector<char, 64> buf;
        getPerfConfigStr(buf);
        return StringAttr::get(getContext(), StringRef(buf.data(), buf.size()));
    }
  }];

  let builders = [AttrBuilderWithInferredContext<(
      ins "StringAttr":$perfConfigStr)>];

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

def Rock_GemmGemmParamsAttr
    : Rock_Attr<"GemmGemmParams", [RockTuningParamAttrInterface]> {
  let mnemonic = "gemm_gemm_params";
  let description = [{
    The tuning parameters for gemm+gemm operation (e.g., attention).

    - mPerBlockG0: The number of values of m to process in each workgroup for
      the first GEMM
    - nPerBlockG0: The number of values of n to process in each workgroup for
      the first GEMM
    - kPerBlock: The number of values of k to process in each workgroup
    - kpack: The number of matrix instructions we can run with one LDS load (can be 1 or 2, see AccelerateAMDMatmul.cpp)
    - numCTAs: The number CTAs
    - numWaves: The number waves
    - matrixInstrNonkdim: The size of the m/n dimension in the accelerator (XDL)
      instruction
    - splitKFactor: Split-k factor for the Split-k GEMM algorithm
    - numStages: Number of stages
    - wavesPerEU: Hint to backend compiler for wavefronts per execution unit
    - gridGroupSize: Number of chiplets to group together to perform a spatially
      local tile
    - useAsyncCopy: Overrides the per-arch default application for the async-copy schedule.
    - useBlockPingpong: Overrides the per-arch default application for the block-pingpong schedule.
    - useInThreadTranspose: Overrides the per-arch default application for the in-thread-transpose pass.
    - useBufferOps: Overrides the convert-to-buffer-ops / canonicalize-pointers / optimize-buffer-op-ptr
      pass cluster.
    - useBufferAtomics: Overrides buffer atomics in convert-to-buffer-ops (requires useBufferOps
      to be enabled).
    - bufferOpsAnalyzeSmallTensorRange: Overrides small-tensor range analysis in
      convert-to-buffer-ops (requires useBufferOps to be enabled).
    - scheduleHint: Bitfield selector mirroring upstream Triton's
      `HIPOptions.schedule_hint`, which is a comma-separated multi-select
      string ("attention,memory-bound-attention"). Encoded as a bitfield so
      that combinations are expressible end-to-end.
      kKnobDefault=-1=arch default, 0=none, bit 0 (=1)=attention,
      bit 1 (=2)=memory-bound-attention. Bit assignments are append-only;
      new variants must claim a new bit and the table in
      KnobUtils.cpp must be extended in lockstep.

    NOTE: The following fields are serialized into the perfConfig but they are
    not actually tuned. Instead, they are always set to `0` (use the backend heuristic)
    or `-1` (the knob default) by the tuner:

    - `wavesPerEU`                       (set to 0)
    - `gridGroupSize`                    (set to 0)
    - `useAsyncCopy`                     (set to -1)
    - `useBlockPingpong`                 (set to -1)
    - `useInThreadTranspose`             (set to -1)
    - `useBufferOps`                     (set to -1)
    - `useBufferAtomics`                 (set to -1)
    - `bufferOpsAnalyzeSmallTensorRange` (set to -1)
  }];

  let parameters = (ins "int64_t":$mPerBlockG0, "int64_t":$nPerBlockG0,
      "int64_t":$kPerBlock, "int64_t":$kpack, "int64_t":$numCTAs,
      "int64_t":$numWaves, "int64_t":$matrixInstrNonkdim,
      "int64_t":$splitKFactor, "int64_t":$numStages, "int64_t":$wavesPerEU,
      "int64_t":$gridGroupSize,
      DefaultValuedParameter<"int64_t", "-1">:$useAsyncCopy,
      DefaultValuedParameter<"int64_t", "-1">:$useBlockPingpong,
      DefaultValuedParameter<"int64_t", "-1">:$useInThreadTranspose,
      DefaultValuedParameter<"int64_t", "-1">:$useBufferOps,
      DefaultValuedParameter<"int64_t", "-1">:$useBufferAtomics,
      DefaultValuedParameter<"int64_t", "-1">:$bufferOpsAnalyzeSmallTensorRange,
      DefaultValuedParameter<"int64_t", "-1">:$scheduleHint);

  let extraClassDeclaration = [{
    void getPerfConfigStr(::llvm::SmallVectorImpl<char> &perfStr) {
      ("attn:v2:"
      + Twine(getMPerBlockG0()) + ","
      + Twine(getNPerBlockG0()) + ","
      + Twine(getKPerBlock()) + ","
      + Twine(getKpack()) + ","
      + Twine(getNumCTAs()) + ","
      + Twine(getNumWaves()) + ","
      + Twine(getMatrixInstrNonkdim()) + ","
      + Twine(getSplitKFactor()) + ","
      + Twine(getNumStages()) + ","
      + Twine(getWavesPerEU()) + ","
      + Twine(getGridGroupSize()) + ","
      + Twine(getUseAsyncCopy()) + ","
      + Twine(getUseBlockPingpong()) + ","
      + Twine(getUseInThreadTranspose()) + ","
      + Twine(getUseBufferOps()) + ","
      + Twine(getUseBufferAtomics()) + ","
      + Twine(getBufferOpsAnalyzeSmallTensorRange()) + ","
      + Twine(getScheduleHint()))
      .toVector(perfStr);
    }

    StringAttr getPerfConfigAttr() {
        SmallVector<char, 64> buf;
        getPerfConfigStr(buf);
        return StringAttr::get(getContext(), StringRef(buf.data(), buf.size()));
    }
  }];

  let builders = [AttrBuilderWithInferredContext<(
      ins "StringAttr":$perfConfigStr)>];

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

def Rock_EnableSplitKForTuning : Rock_Attr<"EnableSplitKForTuning"> {
  let mnemonic = "rock.enable_splitk_for_tuning";
  let description = [{
    Whether we tune for split-k. If unset, split-k=1.
  }];
}

def Rock_Arch : Rock_Attr<"Arch"> {
  let mnemonic = "rock.arch";
  let description = [{
    The target architecture (ISA) string for the module, e.g. "gfx950", "gfx1250".
    This attribute can be attached to modules to specify the target GPU architecture.
  }];
  let parameters = (ins StringRefParameter<"The architecture string">:$value);
  let assemblyFormat = "`<` $value `>`";
}

def Rock_Kernel : Rock_Attr<"Kernel"> {
  let mnemonic = "rock.kernel";
  let description = [{
    Marker attribute to indicate that a function is a GPU kernel.
    This attribute can be attached to func.func operations to mark them as kernels.
  }];
}

def Rock_NumCU : Rock_Attr<"NumCU"> {
  let mnemonic = "rock.num_cu";
  let description = [{
    The number of compute units on the target GPU device. Must be at least the
    per-arch minimum returned by `rock::getMinNumCU` (e.g. gfx906=10,
    gfx908=120, gfx90a=104, gfx942=20, gfx950=256, gfx1010/gfx1030=30,
    gfx1100=2, gfx1200=12, gfx1250=256); any larger value is accepted.
  }];
}

def Rock_NumChiplets : Rock_Attr<"NumChiplets"> {
  let mnemonic = "rock.num_chiplets";
  let description = [{
    The number of chiplets on the target GPU device.
  }];
}

def Rock_BlockSize : Rock_Attr<"BlockSize"> {
  let mnemonic = "rock.block_size";
}

def Rock_GridSize : Rock_Attr<"GridSize"> { let mnemonic = "rock.grid_size"; }

def Rock_ClusterSize : Rock_Attr<"ClusterSize"> {
  let mnemonic = "rock.cluster_size";
}

def Rock_CpuVerifier : Rock_Attr<"CpuVerifier"> {
  let mnemonic = "rock.cpu_verifier";
  let description = [{
    Marker attribute to indicate that a function is a CPU verifier function.
    This attribute is used by buildHostLoweringPipeline to identify functions
    that should be lowered using the CPU verifier pass.
  }];
}

def Rock_HostFunctions : Rock_Attr<"HostFunctions"> {
  let mnemonic = "rock.host_functions";
  let description = [{
    Module attribute carrying serialized host (non-kernel) function bodies
    that `RockSerializeHostFuncsPass` detached so kernel-only Triton/LLVM
    transforms can run, and that `RockEmitGpuBinaryPass` re-parses
    afterwards.  Stored as an `ArrayAttr` of `StringAttr`, where each
    `StringAttr` is the pretty-printed text of one `func.func`.
  }];
}

def Rock_CpuFusedConv : Rock_Attr<"CpuFusedConv"> {
  let mnemonic = "rock.cpu_fused_conv";
  let description = [{
    Marker attribute set on the fused convolution `linalg.generic` produced
    by the `--cpu-conv-to-gemm` pass.
  }];
}

def Rock_NotApplicable : Rock_Attr<"NotApplicable"> {
  let mnemonic = "rock.not_applicable";
  let description = [{
    Marker attribute set on a module by a rock pass when it determines that
    the current (kernel x perf-config x hardware) combination cannot be
    compiled for a structural reason (e.g. required LDS exceeds the arch
    limit, an unsupported hardware feature is needed). The pass should also
    call `signalPassFailure()` so the pipeline halts; consumers (tuning
    driver) inspect this attribute on PM failure to distinguish a
    "not-applicable" config from a real compilation bug.
  }];
}

def Rock_PrefillAttr : Rock_Attr<"Prefill"> {
  let mnemonic = "rock.prefill";

  let description = [{
      A user interface parameter for initializing specific memory buffers

      The attribute is a way to describe which specific kernel parameter
      (tensor) must be initialized with specific value before
      a kernel invocation.

  }];

  let parameters = (ins
    "uint32_t":$argIndex,
    "TypedAttr":$initValue
  );
  let assemblyFormat = "`<` params `>`";
}

#endif
