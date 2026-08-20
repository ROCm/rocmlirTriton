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

/// CacheModifier
///
/// Mirrors triton::CacheModifier (TritonAttrDefs.td) one-to-one (same names and
/// integer values) so a rock cache modifier maps directly onto its Triton
/// counterpart when lowering rock loads to tt.load. Replicated here to keep the
/// Rock dialect free of a TableGen dependency on the Triton dialect.
///
/// The names come from the NVIDIA PTX cache operators (and are reused by Triton
/// on AMD targets). Meanings, as documented by triton.language.load/store:
///   none : no modifier; backend default.
///   ca   : (load) cache at all levels (L1 and L2); the default load behaviour,
///          data is expected to be reused.
///   cg   : cache at global level only (L2 and below, bypassing L1). Valid on
///          both loads and stores.
///   cs   : cache streaming -- data is likely accessed once, so it is treated
///   as
///          evict-first to avoid polluting the cache. Valid on loads and
///          stores.
///   cv   : (load) don't cache and fetch again (treat as volatile /
///   non-coherent
///          read).
///   wb   : (store) write-back to all coherent levels.
///   wt   : (store) write-through to system memory.
/// See the PTX cache operators reference:
/// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
///
/// Loads may use {none, ca, cg, cs, cv}; stores may use {none, cg, cs, wb, wt}.
/// The load-side legality (rejecting the store-only wb/wt) is enforced by
/// verifyLoadCacheModifier() in RockDialect.cpp, not by this enum.

def CacheModifier_None : I32EnumAttrCase<"NONE", 1, "none">;
def CacheModifier_CA : I32EnumAttrCase<"CA", 2, "ca">;
def CacheModifier_CG : I32EnumAttrCase<"CG", 3, "cg">;
def CacheModifier_WB : I32EnumAttrCase<"WB", 4, "wb">;
def CacheModifier_CS : I32EnumAttrCase<"CS", 5, "cs">;
def CacheModifier_WT : I32EnumAttrCase<"WT", 6, "wt">;
def CacheModifier_CV : I32EnumAttrCase<"CV", 7, "cv">;

def CacheModifier
    : Rock_I32Enum<"CacheModifier",
                   "Cache modifier hint for a memory access (mirrors "
                   "triton::CacheModifier)",
                   [CacheModifier_None, CacheModifier_CA, CacheModifier_CG,
                    CacheModifier_WB, CacheModifier_CS, CacheModifier_WT,
                    CacheModifier_CV]>;

def CacheModifierAttr : EnumAttr<Rock_Dialect, CacheModifier, "CacheModifier">;

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

        A Pad may carry the additional marker `tileAlignment`, as in
        ```
          #rock.transform<Pad{0, 15} tileAlignment ["gemmNPad"] at [2] -> ["gemmN"] at [2]>
        ```
        recording that the gemm lowering added this padding to align a gemm
        dimension to the tile size instead of the padding coming from the
        kernel. The lowering masks the matching output lanes off, so this
        padding may be discarded when a load is narrowed to the dimensions it
        actually depends on. Regular padding may not.

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

    let parameters =
        (ins AttrParameter<"::mlir::rock::TransformType",
                           "The type of transformation to perform">:$type,
            ArrayRefParameter<
                "int64_t",
                "The parameters to the transformation, if any">:$params,
            ArrayRefOfStringRefParameter<
                "The names of the upper dimensions (input arguments) of this "
                "transform">:$upperNames,
            ArrayRefParameter<"uint32_t",
                              "The indices of the upper dimensions (input "
                              "arguments) of this transform">:$upperDims,
            ArrayRefOfStringRefParameter<
                "The names of the lower dimensions (outputs) of this "
                "transformation">:$lowerNames,
            ArrayRefParameter<"uint32_t",
                              "The indices of the lower dimensions (outputs) "
                              "of this transformation">:$lowerDims,
            DefaultValuedParameter<"bool", "false",
                                   "Whether this Pad aligns a gemm dimension "
                                   "to the tile size">:$isTileAlignment);

    let builders = [
        // Convenience builder for the transforms that are never tile alignment.
        AttrBuilder<(ins "::mlir::rock::TransformType":$type,
            "::llvm::ArrayRef<int64_t>":$params,
            "::llvm::ArrayRef<::llvm::StringRef>":$upperNames,
            "::llvm::ArrayRef<uint32_t>":$upperDims,
            "::llvm::ArrayRef<::llvm::StringRef>":$lowerNames,
            "::llvm::ArrayRef<uint32_t>":$lowerDims)>,
    ];

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

//===-----------------------------------------------------===//
// Named perfConfig schema
//===-----------------------------------------------------===//

// One field of a named `prefix:key=value,...` perfConfig: the key it is
// serialized under, and the value `get()` falls back to when a config string
// omits that key. The key must match the attribute parameter it describes;
// `Rock_PerfConfigSchema` asserts that at build time.
class RockPerfConfigField<string k, int d> {
  string key = k;
  int defaultValue = d;
}

// The tri-state sentinel every knob field defaults to. This is a copy of
// `rock::kKnobDefault` (Rock/utility/KnobUtils.h) because TableGen cannot read
// a C++ constant; `Rock_PerfConfigSchema` emits a `static_assert` that fails
// the build if the two ever disagree.
defvar kRockKnobDefault = -1;

// An attribute parameter for one tri-state Triton knob: `kKnobDefault`
// (arch/heuristic default), `0` (force off), or `1` (force on). ODS takes the
// default as C++ source text, hence the cast.
def RockKnobParameter
    : DefaultValuedParameter<"int64_t", !cast<string>(kRockKnobDefault)>;

// The perfConfig fields shared by every Rock tuning-param attribute, in
// canonical serialization order. Each attribute prepends its own block-tile
// fields, which are all that differ between the gemm and gemm+gemm schemas.
defvar kRockCommonPerfConfigFields =
    [RockPerfConfigField<"kPerBlock", 16>, RockPerfConfigField<"kpack", 1>,
     RockPerfConfigField<"numCTAs", 1>, RockPerfConfigField<"numWaves", 4>,
     RockPerfConfigField<"matrixInstrNonkdim", 0>,
     RockPerfConfigField<"splitKFactor", 1>,
     RockPerfConfigField<"numStages", 1>, RockPerfConfigField<"wavesPerEU", 0>,
     RockPerfConfigField<"gridGroupSize", 0>,
     RockPerfConfigField<"useAsyncCopy", kRockKnobDefault>,
     RockPerfConfigField<"useBlockPingpong", kRockKnobDefault>,
     RockPerfConfigField<"useInThreadTranspose", kRockKnobDefault>,
     RockPerfConfigField<"useBufferOps", kRockKnobDefault>,
     RockPerfConfigField<"useBufferAtomics", kRockKnobDefault>,
     RockPerfConfigField<"useReductionLayout", kRockKnobDefault>,
     RockPerfConfigField<"useOptimizeEpilogue", kRockKnobDefault>];

// Derives the whole named-perfConfig surface of a tuning-param attribute from a
// single field list: the `getPerfConfigStr()` emitter, the key/default schema
// that `GemmParamsAttr::get(StringAttr)` in RockDialect.cpp parses against, and
// a positional builder pairing with that schema. Adding a field to `params` and
// giving it a default here is therefore enough to have it serialized, accepted
// on parse, and forwarded to `get()` -- none of which can silently drift from
// the others.
//
// `params` must be the same dag the attribute passes to `let parameters`, since
// the emitter calls the accessors ODS derives from it. The assert catches a
// parameter that was added without a matching schema entry (or vice versa),
// which would otherwise silently drop the field from the serialized form.
class Rock_PerfConfigSchema<string prefix, dag params,
                            list<RockPerfConfigField> tileFields> {
  list<RockPerfConfigField> fields =
      !listconcat(tileFields, kRockCommonPerfConfigFields);

  list<string> keys = !foreach(f, fields, f.key);
  list<string> paramNames =
      !foreach(i, !range(0, !size(params)), !getdagname(params, i));

  assert !eq(!interleave(paramNames, ","), !interleave(keys, ",")),
      "perfConfig schema must list exactly this attribute's parameters, in "
      "order: schema has [" # !interleave(keys, ", ") # "], parameters are ["
#!interleave(paramNames, ", ")#"]";

  // `mPerBlock` -> `"mPerBlock=" + Twine(getMPerBlock())`, chained onto the
  // `prefix:` header with `,` separators.
  string emitterExpr =
      "Twine(\"" # prefix # ":\")\n        + "
#!interleave(!foreach(f, fields,
                             "\"" # f.key # "=\" + Twine(get"
#!toupper(!substr(f.key, 0, 1))
#!substr(f.key, 1)#"())"),
                    "\n        + \",\" + ");

  string keysInit =
      "{"#!interleave(!foreach(f, fields, "\""#f.key#"\""), ", ")#"}";
  string defaultsInit = "{"#!interleave(
      !foreach(f, fields, !cast<string>(f.defaultValue)), ", ")#"}";
  string forwardArgs = !interleave(
      !foreach(i, !range(0, !size(fields)), "v["#!cast<string>(i)#"]"), ", ");

  code perfConfigDecls = [{
    // `}]#prefix#[{:key=value,...` is the single canonical output form.
    // Every field is emitted explicitly (defaults can change over time, so we
    // never rely on them being reconstructed on parse). Legacy positional
    // `}]#prefix#[{:vN:` strings remain accepted on input by `get()`.
    //
    // This block is generated from the field list in RockAttrDefs.td; see
    // `Rock_PerfConfigSchema`.

    // `kRockKnobDefault` in RockAttrDefs.td hand-copies the value of
    // `rock::kKnobDefault`, which TableGen cannot read. Emitting that copy as
    // a literal here is what lets C++ compare it against the original.
    static_assert(}]#!cast<string>(kRockKnobDefault)#[{ ==
                      ::mlir::rock::kKnobDefault,
                  "kRockKnobDefault in RockAttrDefs.td is out of sync with "
                  "kKnobDefault in Rock/utility/KnobUtils.h");

    void getPerfConfigStr(SmallVectorImpl<char> &perfStr) {
        (}]#emitterExpr#[{)
        .toVector(perfStr);
    }

    StringAttr getPerfConfigAttr() {
        SmallVector<char, ROCMLIR_TUNING_PARAM_STRING_BUFSZ> buf;
        getPerfConfigStr(buf);
        return StringAttr::get(getContext(), StringRef(buf.data(), buf.size()));
    }

    // The header shared by the named and the legacy positional (`prefix:vN:`)
    // forms.
    static StringRef getPerfConfigPrefix() { return "}]#prefix#[{"; }

    // The keys of the named form in canonical order, and the value each field
    // takes when a config string omits it. Parallel arrays generated from the
    // one field list, so they cannot disagree in length or order.
    static ArrayRef<StringRef> getPerfConfigKeys() {
        static constexpr StringRef keys[] = }]#keysInit#[{;
        return keys;
    }

    static ArrayRef<int64_t> getPerfConfigDefaults() {
        static constexpr int64_t defaults[] = }]#defaultsInit#[{;
        return defaults;
    }

    // Builds the attribute from values ordered as `getPerfConfigKeys()`, which
    // is the order the perfConfig parser resolves them in. The return type is
    // deduced so that this stays independent of the attribute's C++ name.
    static auto getFromPerfConfigValues(MLIRContext *context,
                                        ArrayRef<int64_t> v) {
        assert(v.size() == getPerfConfigKeys().size() &&
               "expected one value per perfConfig field");
        return get(context, }]#forwardArgs#[{);
    }
  }];
}

// Shared by `let parameters` below and the perfConfig schema, so that the
// serialized form and the parser are both derived from this one list.
defvar kRockGemmParams = (ins "int64_t":$mPerBlock, "int64_t":$nPerBlock,
    "int64_t":$kPerBlock, "int64_t":$kpack, "int64_t":$numCTAs,
    "int64_t":$numWaves, "int64_t":$matrixInstrNonkdim, "int64_t":$splitKFactor,
    "int64_t":$numStages, "int64_t":$wavesPerEU, "int64_t":$gridGroupSize,
    RockKnobParameter:$useAsyncCopy, RockKnobParameter:$useBlockPingpong,
    RockKnobParameter:$useInThreadTranspose, RockKnobParameter:$useBufferOps,
    RockKnobParameter:$useBufferAtomics, RockKnobParameter:$useReductionLayout,
    RockKnobParameter:$useOptimizeEpilogue);

def Rock_GemmParamsSchema
    : Rock_PerfConfigSchema<"gemm", kRockGemmParams,
                            [RockPerfConfigField<"mPerBlock", 32>,
                             RockPerfConfigField<"nPerBlock", 32>]>;

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
    - gridGroupSize: Number of M tile blocks grouped together when mapping
      workgroups onto the grid (see `makeGroupedGridLayout`); larger groups
      improve L2 locality across the grouped M-blocks at the cost of
      scheduling flexibility
    - useAsyncCopy: Overrides the per-arch default application for the async-copy schedule.
    - useBlockPingpong: Overrides the per-arch default application for the block-pingpong schedule.
    - useInThreadTranspose: Overrides the per-arch default application for the in-thread-transpose pass.
    - useBufferOps: Overrides the convert-to-buffer-ops / canonicalize-pointers / optimize-buffer-op-ptr
      pass cluster.
    - useBufferAtomics: Overrides buffer atomics in convert-to-buffer-ops (requires useBufferOps
      to be enabled).
    - useReductionLayout: Tri-state gate for the `rock-set-reduction-layout` pass, which
      redistributes GPU warps onto the reduction (K) dim (in convolutions) of the global-load
      gather operand to cut SGPR spill. -1 (heuristic default) rewrites only convolution
      kernels; 0 disables the rewrite entirely; 1 forces it on every kernel.
    - useOptimizeEpilogue: Gate for Triton's `OptimizeEpilogue` pass. `-1`
      selects the rocMLIR store-tail heuristic, `0` disables the pass, and `1`
      enables it unconditionally.

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
    - `useReductionLayout`               (set to -1)
    - `useOptimizeEpilogue`              (set to -1)
  }];

  let parameters = kRockGemmParams;

  let extraClassDeclaration = Rock_GemmParamsSchema.perfConfigDecls;

  let builders = [AttrBuilderWithInferredContext<(
      ins "StringAttr":$perfConfigStr)>];

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

// Shared by `let parameters` below and the perfConfig schema, so that the
// serialized form and the parser are both derived from this one list.
defvar kRockGemmGemmParams = (ins "int64_t":$mPerBlockG0,
    "int64_t":$nPerBlockG0, "int64_t":$nPerBlockG1, "int64_t":$kPerBlock,
    "int64_t":$kpack, "int64_t":$numCTAs, "int64_t":$numWaves,
    "int64_t":$matrixInstrNonkdim, "int64_t":$splitKFactor,
    "int64_t":$numStages, "int64_t":$wavesPerEU, "int64_t":$gridGroupSize,
    RockKnobParameter:$useAsyncCopy, RockKnobParameter:$useBlockPingpong,
    RockKnobParameter:$useInThreadTranspose, RockKnobParameter:$useBufferOps,
    RockKnobParameter:$useBufferAtomics, RockKnobParameter:$useReductionLayout,
    RockKnobParameter:$useOptimizeEpilogue);

// `nPerBlockG1` defaults to 0 ("untiled"), matching how pre-v6 positional
// configs, which have no such field, decode.
def Rock_GemmGemmParamsSchema
    : Rock_PerfConfigSchema<"attn", kRockGemmGemmParams,
                            [RockPerfConfigField<"mPerBlockG0", 32>,
                             RockPerfConfigField<"nPerBlockG0", 32>,
                             RockPerfConfigField<"nPerBlockG1", 0>]>;

def Rock_GemmGemmParamsAttr
    : Rock_Attr<"GemmGemmParams", [RockTuningParamAttrInterface]> {
  let mnemonic = "gemm_gemm_params";
  let description = [{
    The tuning parameters for gemm+gemm operation (e.g., attention).

    - mPerBlockG0: The number of values of m to process in each workgroup for
      the first GEMM
    - nPerBlockG0: The number of values of n to process in each workgroup for
      the first GEMM
    - nPerBlockG1: The number of values of n to process in each workgroup for
      the second GEMM (i.e. the head-dim tile of the attention output). `0`
      means "untiled": the second GEMM processes the full `gemm1N` in one shot
      (the historical behaviour and what pre-v6 positional configs, which have
      no `nPerBlockG1` field, decode to).
    - kPerBlock: The number of values of k to process in each workgroup
    - kpack: The number of matrix instructions we can run with one LDS load (can be 1 or 2, see AccelerateAMDMatmul.cpp)
    - numCTAs: The number CTAs
    - numWaves: The number waves
    - matrixInstrNonkdim: The size of the m/n dimension in the accelerator (XDL)
      instruction
    - splitKFactor: Split-k factor for the Split-k GEMM algorithm
    - numStages: Number of stages
    - wavesPerEU: Hint to backend compiler for wavefronts per execution unit
    - gridGroupSize: Number of M tile blocks grouped together when mapping
      workgroups onto the grid (see `makeGroupedGridLayout`); larger groups
      improve L2 locality across the grouped M-blocks at the cost of
      scheduling flexibility
    - useAsyncCopy: Overrides the per-arch default application for the async-copy schedule.
    - useBlockPingpong: Overrides the per-arch default application for the block-pingpong schedule.
    - useInThreadTranspose: Overrides the per-arch default application for the in-thread-transpose pass.
    - useBufferOps: Overrides the convert-to-buffer-ops / canonicalize-pointers / optimize-buffer-op-ptr
      pass cluster.
    - useBufferAtomics: Overrides buffer atomics in convert-to-buffer-ops (requires useBufferOps
      to be enabled).
    - useReductionLayout: Tri-state gate for the `rock-set-reduction-layout` pass, which
      redistributes GPU warps onto the reduction (K) dim (in convolutions) of the global-load
      gather operand to cut SGPR spill. -1 (heuristic default) rewrites only convolution
      kernels; 0 disables the rewrite entirely; 1 forces it on every kernel.
    - useOptimizeEpilogue: Gate for Triton's `OptimizeEpilogue` pass. `-1`
      selects the rocMLIR store-tail heuristic, `0` disables the pass, and `1`
      enables it unconditionally.

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
    - `useReductionLayout`               (set to -1)
    - `useOptimizeEpilogue`              (set to -1)
  }];

  let parameters = kRockGemmGemmParams;

  let extraClassDeclaration = Rock_GemmGemmParamsSchema.perfConfigDecls#[{
    // Bridge accessors for the RockTuningParamAttrInterface so generic code
    // (e.g. perf-config validation) can treat gemm and gemm+gemm uniformly.
    int64_t getMPerBlock() const { return getMPerBlockG0(); }
    int64_t getNPerBlock() const { return getNPerBlockG0(); }
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

def Rock_ConvKernel : Rock_Attr<"ConvKernel"> {
  let mnemonic = "rock.conv_kernel";
  let description = [{
    Marker attribute indicating that a kernel function was lowered from a
    convolution (`rock.conv`, `rock.conv_bwd_data`, `rock.conv_bwd_weight`, or
    `rock.conv_elementwise_gemm`).
  }];
}

def Rock_NumCU : Rock_Attr<"NumCU"> {
  let mnemonic = "rock.num_cu";
  let description = [{
    The number of compute units on the target GPU device. Must be at least the
    per-arch minimum returned by `rock::getMinNumCU` (e.g. gfx906=10,
    gfx908=120, gfx90a=104, gfx942=20, gfx950=128, gfx1010/gfx1030=30,
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

def Rock_UseOptimizeEpilogue : Rock_Attr<"UseOptimizeEpilogue"> {
  let mnemonic = "rock.use_optimize_epilogue";
  let description = [{
    Temporary function attribute carrying the perfConfig tri-state from
    `rock-affix-params` to `rock-add-triton-metadata`. The metadata pass removes
    it after resolving the automatic policy to `rock.prefer_lds_epilogue`, so
    raw tuning policy does not enter Triton IR.
  }];
}

def Rock_PreferLdsEpilogue : Rock_Attr<"PreferLdsEpilogue"> {
  let mnemonic = "rock.prefer_lds_epilogue";
  let description = [{
    Marker set on a kernel function by `rock-add-triton-metadata` when the
    store-tail heuristic prefers the paced LDS epilogue. The marker survives
    TTIR lowering and is consumed by Triton's OptimizeEpilogue pass to retain
    that epilogue.
  }];
}

def Rock_GridSize : Rock_Attr<"GridSize"> {
  let mnemonic = "rock.grid_size";

  let extraClassDeclaration = [{
    static ::std::string getModuleAttrName(::llvm::StringRef kernelName) {
      return (::llvm::Twine(getMnemonic()) + "." + kernelName).str();
    }
  }];
}

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

def Rock_OTransposed : Rock_Attr<"OTransposed"> {
  let mnemonic = "o_transposed";

  let description = [{
    Discardable metadata describing the in-memory layout of a GEMM output
    tile.  It is attached (as a discardable attribute named `rock.o_transposed`)
    to `rock.blockwise_gemm` by the `rock-add-triton-metadata` pass and is
    carried onto the lowered `tt.dot` / `tt.dot_scaled` (and survives the
    Triton `tritonamdgpu-accelerate-matmul` rewrite).

    - `value = true`  -> the output is "transposed" (column-major: the M
                         dimension is the fast/contiguous dimension of the
                         output in global memory).
    - `value = false` -> the output is row-major (the N dimension is fast).

    `rock-set-matmul-output-transpose` later consumes this to pick the
    accelerator result layout (e.g. the WMMA `isTransposed` flag) so the
    epilogue store coalesces/vectorizes.
  }];

  let parameters = (ins "bool":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    /// Name of the discardable attribute used to carry this metadata on ops
    /// (e.g. `rock.blockwise_gemm`, `tt.dot`, `tt.dot_scaled`).
    static constexpr ::llvm::StringLiteral getNameStr() {
      return ::llvm::StringLiteral("rock.o_transposed");
    }
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
