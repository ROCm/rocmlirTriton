//===- RockOps.td - Rock operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Rock operations.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_OPS
#define ROCK_OPS

include "mlir/Dialect/Rock/IR/RockAttrDefs.td"
include "mlir/Dialect/Rock/IR/RockConvInterface.td"
include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockAcceptingViewOpInterface.td"
include "mlir/Dialect/Rock/IR/RockWriterOpInterface.td"
include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for Rock dialect ops.
class Rock_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rock_Dialect, mnemonic, traits> {
  let extraClassDeclaration = [{
  }];
}

def TransformMapArrayAttr : TypedArrayAttrBase<Rock_TransformMapAttr,
  "Coordinate transforms array attribute, giving the sequence of transform maps applicable to a value, uppermost to lowermost"> {}

def ArgTransformsAttr : TypedArrayAttrBase<TransformMapArrayAttr,
  "An array of arrays of transformations, one per operation argument">;

class ArgTransforms<int n> : ConfinedAttr<ArgTransformsAttr, [ArrayCount<n>]>;

class IndexArrayLength<int n> : ConfinedAttr<IndexArrayAttr, [ArrayMinCount<n>]>;

defvar commonConvAttrs = (ins IndexArrayLength<4>:$padding,
    IndexArrayLength<2>:$strides, IndexArrayLength<2>:$dilations,
    OptionalAttr<RockTuningParamAttrInterface>:$params);

class Rock_ConvOpBase<string mnemonic>
    : Rock_Op<mnemonic, [DeclareOpInterfaceMethods<RockGemmWrapperInterface>,
                         DeclareOpInterfaceMethods<RockConvInterface>,
                         RockFusionRoot]> {
  let results = (outs AnyRankedTensor:$result);
}

defvar GemmInputTypes = [F32, F16, BF16, I8, F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2,
                         F8E4M3FN, F4E2M1FN, F8E8M0FNU];
// This can be extended for quantization.
defvar GemmOutputTypes = [F32, F16, BF16, I32, I8, F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2, F8E4M3FN];
defvar GemmAccumulatorTypes = [F32, F16, BF16, I32];

def Rock_ConvOp : Rock_ConvOpBase<"conv"> {
  let arguments = !con((ins RankedTensorOf<GemmInputTypes>:$filter,
                           RankedTensorOf<GemmInputTypes>:$input),
                       commonConvAttrs);
  let summary = "N-D convolution forward";
  let description = [{
    The `rock.conv` op computes N-D forward convolution.
    Takes filter and input tensors, produces the output feature map as $result.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` attr-dict
    `:` type(operands) `->` type($result)
  }];
}

def Rock_ConvBwdDataOp : Rock_ConvOpBase<"conv_bwd_data">
{
  let arguments = !con((ins RankedTensorOf<GemmInputTypes>:$filter,
                           RankedTensorOf<GemmInputTypes>:$gradient),
                       commonConvAttrs);
  let summary = "N-D convolution backward data";
  let description = [{
    The `rock.conv_bwd_data` op computes N-D convolution backward data.
    Takes filter and gradient (from the forward output) tensors, produces the
    gradient with respect to the input as $result.

    Multiple implicit GEMM kernels may be needed for the correct computation
    of the result (e.g. when stride > 1). The ConvToGemm pass expands a single
    `conv_bwd_data` into the required set of GEMM + store pairs within the
    same function.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` attr-dict
    `:` type(operands) `->` type($result)
  }];
}


def Rock_ConvBwdWeightOp : Rock_ConvOpBase<"conv_bwd_weight">
{
  let arguments = !con((ins RankedTensorOf<GemmInputTypes>:$input,
                           RankedTensorOf<GemmInputTypes>:$gradient),
                       commonConvAttrs,
                       (ins Optional<RankedTensorOf<[F32]>>:$workspace,
                           OptionalAttr<IndexAttr>:$kBlocks));
  let summary = "N-D convolution backward weight";
  let description = [{
    The `rock.conv_bwd_weight` op computes N-D convolution backward weight.
    Takes input and gradient (from the forward output) tensors, produces the
    gradient with respect to the filter weights as $result.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` attr-dict
    `:` type(operands) `->` type($result)
  }];
}

def Rock_GemmOp
    : Rock_Op<"gemm", [DeclareOpInterfaceMethods<RockGemmWrapperInterface>,
                       RockFusionRoot, AttrSizedOperandSegments]>,
      Arguments<(ins Arg<TensorOf<GemmInputTypes>, "matrix A">:$a,
          Arg<TensorOf<GemmInputTypes>, "matrix B">:$b,
          Optional<TensorOf<[F8E8M0FNU]>>:$scaleA,
          Optional<TensorOf<[F8E8M0FNU]>>:$scaleB, UnitAttr:$aTransposed,
          UnitAttr:$bTransposed, UnitAttr:$oTransposed,
          UnitAttr:$aScaleTransposed, UnitAttr:$bScaleTransposed,
          OptionalAttr<I64Attr>:$quantBlockSize,
          OptionalAttr<RockTuningParamAttrInterface>:$params)>,
      Results<(outs AnyRankedTensor:$c)> {
  let summary = "General matrix multiplication (GEMM)";
  let description = [{
    Performs the operation c = A * B.

    If none of the `transposed` attributes are set, then A is [G] x M x K,
    B is [G] x K x N, and c is [G] x M x N, where G is the optional group
    dimension (which is assumed to be 1 if not set).

    The transpose attributes allow for the non-group dimensions of the matrix to be
    transposed. For example, if `aTransposed` is set, then the argument A should be
    a [G] x K x M memory. Note that `oTransposed` transposes the result of this op.

    When `scaleA` and `scaleB` are present, this op performs a block-scaled GEMM
    (MXFP-style). The `quantBlockSize` attribute specifies the number of
    consecutive elements along the K dimension that share a single scale factor.
    The scale tensors have shape [G] x M x ceil(K / quantBlockSize) for `scaleA`
    and [G] x N x ceil(K / quantBlockSize) for `scaleB`.
    The scale type should be f8E8M0FNU (E8M0), where each byte encodes a
    power-of-two exponent: value = 2^(byte - 127). This matches the MXFP block
    scaling convention used by hardware instructions such as
    V_MFMA_SCALE_F32_*_F8F6F4 on AMD gfx950.

    The parameters `derivedBlockSize`, `gridSize`, and `params` are optional as
    they can be inferred by a tuning process or a heuristic, but they must be
    set before the `gemm` is lowered into the `gridwise_gemm` stage of the code
    generation pipeline.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    (`tr` $aTransposed^)? $a (`scaled` `by` (`tr` $aScaleTransposed^)? $scaleA^)? `*` (`tr` $bTransposed^)? $b (`scaled` `by` (`tr` $bScaleTransposed^)? $scaleB^)?
    attr-dict
    `:` type($a) (`scaled` `by` type($scaleA)^)? `*` type($b) (`scaled` `by` type($scaleB)^)? `->` type($c)
  }];
}

def Rock_ReduceOp
    : Rock_Op<"reduce", [AllElementTypesMatch<["in", "result"]>]>,
      Arguments<(ins Arg<TensorOf<[BF16, F16, F32]>, "input">:$in,
          ReduceMethodAttr:$reduceMethod, IndexAttr:$axis)>,
      Results<(outs TensorOf<[BF16, F16, F32]>:$result)> {
  let summary = "Axes-wide reduction operation";
  let hasVerifier = 1;
  let assemblyFormat = [{
    `` $reduceMethod $in attr-dict `:` type($in) `->` type($result)
  }];
}

def Rock_AttentionOp
    : Rock_Op<
          "attention", [DeclareOpInterfaceMethods<RockGemmGemmWrapperInterface>,
                        RockFusionRoot, AttrSizedOperandSegments]>,
      Arguments<(ins TensorOf<[F32, F16, BF16, I8]>:$queries,
          TensorOf<[F32, F16, BF16, I8]>:$keys,
          TensorOf<[F32, F16, BF16]>:$values,
          Variadic<AnyTensor>:$preSoftmaxElemWiseInputs,
          Optional<TensorOf<[I32]>>:$currentSeqLen,
          Optional<TensorOf<[I32]>>:$prefixOffset, I32Attr:$numHeadsQ,
          I32Attr:$numHeadsKV, UnitAttr:$qTransposed, UnitAttr:$kTransposed,
          UnitAttr:$vTransposed, UnitAttr:$oTransposed, UnitAttr:$causal,
          I32Attr:$splitKV, OptionalAttr<TypeAttr>:$softmaxType,
          OptionalAttr<RockTuningParamAttrInterface>:$params0,
          OptionalAttr<RockTuningParamAttrInterface>:$params1,
          DenseI64ArrayAttr:$firstGemmIndices,
          DefaultValuedOptionalAttr<BoolAttr,
                                    "false">:$preSoftmaxHasSplitKVTransforms)>,
      Results<(outs TensorOf<[F32, F16, BF16]>:$result,
          Optional<TensorOf<[F32, F16, BF16]>>:$lse)> {
  let summary = "Attention operation of transformer models";
  let description = [{
    Performs the operation out = SOFTMAX(preSoftmaxBody(queries * keys, preSoftmaxElemWiseInputs)) * values.

    This operation performs attention mechanism of transformer models. There is an optional element-wise 
    fusion just before the softmax, defined by `preSoftmaxBody` with inputs `preSoftmaxElemWiseInputs`.

    If none of the `transposed` attributes are set, then `queries` is [G] x seq_q x head_qk,
    `keys` is [G] x head_qk x seq_k, `values` is [G] x seq_k x head_v and `out` is [G] x seq_q x head_v, 
    where G is the optional group dimension (which is assumed to be 1 if not set).

    The transpose attributes allow for the non-group dimensions of the matrix to be
    transposed. For example, if `qTransposed` is set, then the argument `queries` should be
    a [G] x head_qk x seq_q memory.

    If causal is enabled, we implement causal masking.

    If causal is enabled AND prefixOffset is not null, we implement prefix
    causal masking where: mask = key_pos > (query_pos + offset).
    The prefixOffset can be either:
    - A scalar (shape [1]): a single global offset applied to all sequences
    - A tensor of shape [G]: per-group/batch offsets, allowing different prefix
      lengths for each sequence in the batch

    LSE (log-sum-exp) is an optional output typically used for flash decoding.
    For flash decoding, you can pass splitKV > 1, the default value is 1, which means flash decoding is disabled. 
    Flash decoding multiplies the number of blocks by splitKV. Note that "lse" has to be non-null for splitKV > 1. 
    However, "lse" can be non-null even if splitKV = 1.

    The `preSoftmaxHasSplitKVTransforms` attribute indicates whether the preSoftmaxBody inputs 
    have splitKV transforms applied. It is set to true by DetectFlashDecoding when it detects 
    and removes splitKV patterns. This prevents GridwiseGemmToBlockwise from unnecessarily 
    re-applying splitKV transforms when the IR is generated directly (e.g., via rocmlir-gen) 
    without flash decoding detection.

    The parameters `gridSize`, and `blockSize` are optional as they can be
    inferred by a tuning process or a heuristic, but they must be set before
    the `attention` is lowered into the `gridwise_attention` stage of the
    code generation pipeline.
  }];
  let hasVerifier = 1;
  let regions = (region AnyRegion:$preSoftmaxBody);
  let assemblyFormat = [{
    `{` `\n`
        ` ` `qk` `=` (`tr` $qTransposed^)? $queries `*` (`tr` $kTransposed^)? $keys `:` type($queries) `,` type($keys) `\n`
        (`currentSeqLen` `=` `(` $currentSeqLen^ `:` type($currentSeqLen) `)` `\n`)?
        (`prefixOffset` `=` `(` $prefixOffset^ `:` type($prefixOffset) `)` `\n`)?
        (`causal` `\n` $causal^)?
        (`qk` `=` `elementwise` (`otherIns` `(` $preSoftmaxElemWiseInputs^ `:` type($preSoftmaxElemWiseInputs) `)`)? $preSoftmaxBody^ `\n`)?
        `softmax` `(` `qk` `)` `*` (`tr` $vTransposed^)? $values `:` type($values) `\n`
    `}` attr-dict `->` type($result) (`,` type($lse)^)?
  }];
}

def Rock_GemmElementwiseGemmOp
    : Rock_Op<"gemm_elementwise_gemm", [DeclareOpInterfaceMethods<
                                            RockGemmGemmWrapperInterface>,
                                        RockFusionRoot]>,
      AllElementTypesMatch<["a", "b"]>,
      Arguments<(ins TensorOf<[F32, F16, BF16]>:$a,
          TensorOf<[F32, F16, BF16]>:$b, TensorOf<[F32, F16, BF16]>:$c,
          Variadic<AnyTensor>:$elemwiseInputs, UnitAttr:$aTransposed,
          UnitAttr:$bTransposed, UnitAttr:$cTransposed, UnitAttr:$oTransposed,
          OptionalAttr<RockTuningParamAttrInterface>:$params0,
          OptionalAttr<RockTuningParamAttrInterface>:$params1,
          DenseI64ArrayAttr:$firstGemmIndices)>,
      Results<(outs TensorOf<[F32, F16, BF16]>:$result)> {
  let summary = "GEMM-elementwise-GEMM operation";
  let description = [{
    Performs the operation out = preSecondGemmBody(a * b, elemwiseInputs) * c.

    This operation performs fused GEMM-elementwise-GEMM. There is an optional element-wise
    fusion just before the second GEMM, defined by `preSecondGemmBody` with inputs `elemwiseInputs`.

    If none of the `transposed` attributes are set, then `a` is [G] x M x K,
    `b` is [G] x K x N, `c` is [G] x N x O and `out` is [G] x M x O, where G is the 
    optional group dimension (which is assumed to be 1 if not set).

    The transpose attributes allow for the non-group dimensions of the matrix to be
    transposed. For example, if `aTransposed` is set, then the argument `a` should be
    a [G] x K x M memory.

    The parameters `gridSize`, and `blockSize` are optional as they can be
    inferred by a tuning process or a heuristic, but they must be set before
    the `gemm_elementwise_gemm` is lowered into the `gridwise_attention`
    stage of the code generation pipeline.
  }];
  let hasVerifier = 1;
  let regions = (region AnyRegion:$preSecondGemmBody);
  let assemblyFormat = [{
    `{` `\n`
        ` ` `ab` `=` (`tr` $aTransposed^)? $a `*` (`tr` $bTransposed^)? $b `:` type($a) `,` type($b) `\n`
        (`ab` `=` `elementwise` (`otherIns` `(` $elemwiseInputs^ `:` type($elemwiseInputs) `)`)? $preSecondGemmBody^ `\n`)?
        `ab` `*` (`tr` $cTransposed^)? $c `:` type($c) `\n`
    `}` attr-dict `->` type($result)
  }];
}

def Rock_ConvElementwiseGemmOp
    : Rock_Op<"conv_elementwise_gemm", [DeclareOpInterfaceMethods<
                                            RockGemmGemmWrapperInterface>,
                                        RockFusionRoot]>,
      AllElementTypesMatch<["filter", "input"]>,
      Arguments<(ins TensorOf<[F32, F16, BF16]>:$filter,
          TensorOf<[F32, F16, BF16]>:$input, TensorOf<[F32, F16, BF16]>:$c,
          Variadic<AnyTensor>:$elemwiseInputs, TensorOf<[F32, F16, BF16]>:$out,
          UnitAttr:$cTransposed, UnitAttr:$oTransposed,
          IndexArrayLength<4>:$padding, IndexArrayLength<2>:$strides,
          IndexArrayLength<2>:$dilations,
          OptionalAttr<RockTuningParamAttrInterface>:$params0,
          OptionalAttr<RockTuningParamAttrInterface>:$params1,
          DenseI64ArrayAttr:$firstGemmIndices)>,
      Results<(outs Optional<TensorOf<[F32, F16, BF16]>>:$result)> {
  let summary = "CONV-elementwise-GEMM operation";
  let description = [{
    Performs the operation out = preSecondGemmBody(conv_forward(filter, input), elemwiseInputs) * c.

    This operation performs fused CONV-elementwise-GEMM. There is an optional element-wise
    fusion just before the second GEMM, defined by `preSecondGemmBody` with inputs `elemwiseInputs`.

    If none of the `transposed` attributes are set, then `c` is [G] x N x O and `out` is [G] x M x O, 
    where G is the optional group dimension (which is assumed to be 1 if not set). Note that input and
    filter layouts are defined by attributes "input_layout" and "filter_layout".

    The transpose attributes allow for the non-group dimensions of the matrix to be
    transposed. For example, if `cTransposed` is set, then the argument `c` should be
    a [G] x O x M memory.

    The parameters `gridSize`, and `blockSize` are optional as they can be
    inferred by a tuning process or a heuristic, but they must be set before
    the `conv_elementwise_gemm` is lowered into the `gridwise_attention`
    stage of the code generation pipeline.
  }];
  let hasVerifier = 1;
  let regions = (region AnyRegion:$preSecondGemmBody);
  let assemblyFormat = [{
    `{` `\n`
        ` ` `ab` `=` `conv` `(` $filter `,` $input `)` `:` type($filter) `,` type($input) `\n`
        (`ab` `=` `elementwise` (`otherIns` `(` $elemwiseInputs^ `:` type($elemwiseInputs) `)`)? $preSecondGemmBody^ `\n`)?
        (`tr` $oTransposed^)? $out `=` `ab` `*` (`tr` $cTransposed^)? $c `:` type($c) `->` type($out) `\n`
    `}` attr-dict (`->` type($result)^)?
  }];
}

def Rock_TransformOp :
    Rock_Op<"transform", [Pure, ViewLikeOpInterface,
                          AllElementTypesMatch<["input", "output"]>]>,
    Arguments<(ins AnyShaped:$input, Rock_TransformMapAttr:$transform)>,
    Results<(outs AnyShaped:$output)> {
  let summary = "Tensor transformation";
  let description = [{
    Create a view `output` of the tensor `input` with the same element type
    such that, when the coordinates used to index into `output` are passed through
    the coordinate transformations `transforms` from left to right,
    they become coordinates into the `input` tensor.
  }];
  let builders = [
   // Custom builder to populate bounds of input and output tensors as attributes.
   OpBuilder<(ins "Value":$input, "TransformMapAttr":$transform),
   [{
     $_state.addOperands({input});
     ShapedType lowerType = cast<ShapedType>(input.getType());
     Type elemType = lowerType.getElementType();
     ShapedType upperType = lowerType.cloneWith(transform.getUpperBounds().asArrayRef(), elemType);
     $_state.addAttribute("transform", transform);
     $_state.addTypes(upperType);
   }]>,
  ];
  let hasVerifier = 1;
  let assemblyFormat = [{
    $input `by` $transform attr-dict `:` type($input) `to` type($output)
  }];

  let extraClassDeclaration = [{
    Value getViewSource() { return getInput(); }
  }];
}

// untile
def Rock_UntileOp : Rock_Op<"untile",
                            [Pure, AllElementTypesMatch<["source", "result"]>]>,
                    Arguments<(ins AnyRankedTensor:$source)>,
                    Results<(outs AnyRankedTensor:$result)> {
  let summary = "Convert a tile tensor back to a full tensor type";
  let description = [{
    The rock.untile op maps a tile tensor (e.g., from a BlockwiseLoadOp) to a
    full tensor type. It is used in InsertOutputFusionLoads to bridge loaded
    tiles back to full tensor types so that fusion ops can operate on them.
    LowerStores removes these ops when converting back to tile-level operations.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

// store_marker
def Rock_StoreMarkerOp
    : Rock_Op<"store_marker",
              [Pure, AllElementTypesMatch<["source", "result"]>]>,
      Arguments<(ins AnyRankedTensor:$source, TransformMapArrayAttr:$extraViews,
          Variadic<I32>:$extraIndices)>,
      Results<(outs AnyRankedTensor:$result)> {
  let summary =
      "Mark a tile tensor with output transforms and map to full tensor type";
  let description = [{
    The rock.store_marker op marks a tile tensor (e.g., from a GEMM loop result)
    with the transform information needed to write it to global memory, and
    produces the corresponding full tensor type. This op is a placeholder that:
    1. Maps from tile type (source) to full tensor type (result)
    2. Carries the output transforms (extraViews) computed during ToBlockwise
    3. Carries the grid coordinates (extraIndices) for the tile position

    In LowerStores, this op is combined with rock.store to create the final
    rock.blockwise_store with proper transform chains.

    Example:
      %tile = scf.for ... -> tensor<16x16xf32>  // GEMM loop result
      %full = rock.store_marker %tile views [<transforms>] [%g, %m, %n]
                : tensor<16x16xf32> -> tensor<1x128x128xf32>
      %fused = arith.addf %full, %bias : tensor<1x128x128xf32>
      %out = rock.store %fused to %dest by set
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `views` $extraViews (`[` $extraIndices^ `]`)? attr-dict
    `:` type($source) `->` type($result)
  }];
}

// load_marker
def Rock_LoadMarkerOp
    : Rock_Op<"load_marker",
              [Pure, AllElementTypesMatch<["source", "result"]>]>,
      Arguments<(ins AnyRankedTensor:$source, TransformMapArrayAttr:$extraViews,
          Variadic<I32>:$extraIndices)>,
      Results<(outs AnyRankedTensor:$result)> {
  let summary = "Mark a tensor with tiling transforms for later load lowering";
  let description = [{
    The rock.load_marker op is the dual of rock.store_marker. It marks a source
    tensor (which may be the result of transforms and/or fusion ops tracing back
    to a block argument) with the tiling information needed to create an actual
    rock.blockwise_load.

    This op is a placeholder introduced by:
    1. ToBlockwise passes (GridwiseGemmToBlockwise, GridwiseAttnToBlockwise)
       for the GEMM/Attention input loads. In this case, the source is the
       GEMM input (possibly through a fusion chain), extraViews are the tiling
       transforms, extraIndices are grid coordinates, and the result type is
       the tile type (consumed by rock.blockwise_gemm).
    2. InsertOutputFusionLoads for extra fusion inputs in the output chain.
       In this case, the source is the extra input, extraViews are the output
       tiling transforms, extraIndices are grid coordinates, and the result
       type is the full tensor type (consumed by fusion ops).

    LowerLoads converts this op into an actual rock.blockwise_load by tracing
    back through the source chain (transforms, fusion ops, block argument)
    and combining all transforms.

    Example (GEMM input, tile result):
      %a = rock.transform %arg0 by <pre_transforms>
      %tile = rock.load_marker %a views [<tiling>] [%kIter, %g, %m, %n]
                : tensor<1x128x128xf16> -> tensor<64x64xf16>

    Example (output fusion extra input, full result):
      %bias_t = rock.transform %arg_bias by <transforms>
      %bias = rock.load_marker %bias_t views [<output_tiling>] [%g, %m, %n]
                : tensor<1x128x128xf16> -> tensor<1x128x128xf16>
      %fused = arith.addf %gemm_full, %bias
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `views` $extraViews (`[` $extraIndices^ `]`)? attr-dict
    `:` type($source) `->` type($result)
  }];
}

// store
def Rock_StoreOp : Rock_Op<"store",
                           [AllElementTypesMatch<["source", "dest", "result"]>]>,
                   Arguments<(ins AnyRankedTensor:$source,
                       AnyRankedTensor:$dest, StoreMethodAttr:$storeMethod)>,
                   Results<(outs AnyRankedTensor:$result)> {
  let summary = "Store tensor result to destination";
  let description = [{
    The rock.store op is a placeholder marking where stores to global memory need to happen. 
    This op is used to specify that the output of a given operation (i.e., rock.gemm) will be stored into 
    a different (tensor) location. This is needed because we have rock.transform ops that are used to 
    modify the way we load/store to memory, whereas ops like rock.gemm return a tensor result which do 
    not take into account the rock.transform ops, so rock.store ops allow us to indicate that writing to 
    the output will occur according to what the rock.transform op indicates.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `to` $dest `by` $storeMethod attr-dict `:` type($source) `->` type ($result) `to` type($dest)
  }];
}

// gridwise_gemm
def Rock_GridwiseGemmOp
    : Rock_Op<"gridwise_gemm", [RockFusionRoot, AttrSizedOperandSegments]>,
      Arguments<(ins Arg<TensorRankOf<GemmInputTypes, [3]>, "matrix A view">:$a,
          Arg<TensorRankOf<GemmInputTypes, [3]>, "matrix B view">:$b,
          Optional<TensorRankOf<[F8E8M0FNU], [3]>>:$scaleA,
          Optional<TensorRankOf<[F8E8M0FNU], [3]>>:$scaleB,
          OptionalAttr<I64Attr>:$quantBlockSize, Rock_GemmParamsAttr:$params)>,
      Results<(outs AnyRankedTensor:$c)> {
  let summary = "Gridwise GEMM";
  let description = [{
    The `rock.gridwise_gemm` op computes gridwise GEMM.

    When `scaleA` and `scaleB` are present, this op performs a block-scaled GEMM.
    The `quantBlockSize` attribute specifies the number of consecutive elements
    along the K dimension that share a single scale factor. Each scale tensor
    has its K dimension reduced by a factor of `quantBlockSize` relative to the
    corresponding data tensor (i.e., kScale = ceil(K / quantBlockSize)).
    The scale type should be f8E8M0FNU (E8M0), encoding a power-of-two
    exponent per block: value = 2^(byte - 127).
  }];
  let assemblyFormat = [{
    `(` $a `,` $b
        (`,` $scaleA^)? (`,` $scaleB^)? `)`
    attr-dict
    `:` type($a) `,` type($b)
       (`,` type($scaleA)^)? (`,` type($scaleB)^)?
    `->` type($c)
  }];
  let hasVerifier = 1;
}

// gridwise_attention
def Rock_GridwiseAttentionOp
    : Rock_Op<"gridwise_attention", [RockFusionRoot, AttrSizedOperandSegments,
]>,
      Arguments<(ins TensorRankOf<[F32, F16, BF16, I8], [3]>:$queries,
          TensorRankOf<[F32, F16, BF16, I8], [3]>:$keys,
          TensorRankOf<[F32, F16, BF16], [3]>:$values,
          Variadic<AnyRankedTensor>:$preSoftmaxElemWiseInputs,
          Optional<TensorRankOf<[I32], [1]>>:$currentSeqLen,
          Optional<TensorRankOf<[I32], [1]>>:$prefixOffset, UnitAttr:$causal,
          I32Attr:$splitKV, UnitAttr:$disableQBypassLDS,
          OptionalAttr<IndexAttr>:$prePadG0M,
          OptionalAttr<IndexAttr>:$prePadG0N,
          OptionalAttr<IndexAttr>:$numRepeatsGQA,
          OptionalAttr<TypeAttr>:$softmaxType, Rock_GemmParamsAttr:$params0,
          Rock_GemmParamsAttr:$params1, DenseI64ArrayAttr:$firstGemmIndices,
          DefaultValuedOptionalAttr<BoolAttr, "true">:$enableSoftmax,
          DefaultValuedOptionalAttr<BoolAttr,
                                    "false">:$preSoftmaxHasSplitKVTransforms)>,
      Results<(outs TensorOf<[F32, F16, BF16]>:$result,
          Optional<TensorOf<[F32, F16, BF16]>>:$lse)> {
  let summary = "Gridwise attention";
  let description = [{
    The `rock.gridwise_attention` op computes gridwise attention.
  }];
  let regions = (region AnyRegion:$preSoftmaxBody);
  let assemblyFormat = [{
    `(` operands `)` `preSoftmaxOps` `=` $preSoftmaxBody attr-dict `:` type(operands)
    `->` type($result) (`,` type($lse)^)?
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Block arg 0 of preSoftmaxBody is the first GEMM output (QK product).
    BlockArgument getPreSoftmaxQKArgument() {
      Region &region = getPreSoftmaxBody();
      if (region.empty())
        return nullptr;
      return region.front().getArgument(0);
    }
  }];
}

// yield
def Rock_YieldOp :
    Rock_Op<"yield", [Pure, Terminator, ReturnLike]> {
  let summary = "yield for Rock loops";

  let description = [{
    Op that terminates the Rock looping constructs (currently transforming for).

    If the loops produce any values, then this op's arguments should match the types
    of those values. Otherwise, this op is implicit and can be omitted.

    Note that, in a transforming_for, you don't need to yield back the lower coordinates,
    as these are managed implicitly.
  }];

  // Yoinked from affine.yield
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins),
                            [{ build($_builder, $_state, ValueRange{}); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

defvar SupportedMemoryElems = [F32, F16, BF16, I1, I<4>, I8, I16, I32,
                               F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2, F8E4M3FN,
                               F4E2M1FN, F8E8M0FNU];
defvar NativeMemoryOpTypes =
    [F32, F16, BF16, I<4>, I8, I16, I32, F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2,
     F8E4M3FN, F4E2M1FN, F8E8M0FNU,
     VectorOfLengthAndType<[2, 4, 8, 16, 32], [F32, I32, F4E2M1FN, F8E8M0FNU]>,
     VectorOfLengthAndType<[2, 4, 8, 16], [F16, BF16]>,
     VectorOfLengthAndType<[2, 4, 8, 16, 32, 64], [I8, F8E5M2FNUZ, F8E4M3FNUZ,
                                                   F8E5M2, F8E4M3FN]>];

defvar LdsBufferTypes = GemmInputTypes # [VectorOfRankAndType<[1], GemmInputTypes>];

// blockwise_gemm
def Rock_BlockwiseGemmOp
    : Rock_Op<"blockwise_gemm", [AttrSizedOperandSegments]>,
      Arguments<(ins TensorOf<GemmInputTypes>:$matrixA,
          TensorOf<GemmInputTypes>:$matrixB, TensorOf<GemmOutputTypes>:$matrixC,
          Optional<TensorOf<[F8E8M0FNU, I8]>>:$matrixScaleA,
          Optional<TensorOf<[F8E8M0FNU, I8]>>:$matrixScaleB,
          OptionalAttr<I64Attr>:$quantBlockSize,
          OptionalAttr<TypeAttr>:$matrixAOrigElemType,
          OptionalAttr<TypeAttr>:$matrixBOrigElemType,
          OptionalAttr<BoolAttr>:$matrixAKPack,
          OptionalAttr<BoolAttr>:$matrixBKPack)>,
      Results<(outs TensorOf<GemmOutputTypes>:$result)> {
  let summary = "Blockwise GEMM";
  let description = [{
    The `rock.blockwise_gemm` op does GEMM at workgroup (block) level.

    When `matrixScaleA` and `matrixScaleB` are present, this op performs a
    block-scaled GEMM. The `quantBlockSize` attribute specifies the number of
    consecutive elements along the K dimension that share a single scale factor.
    Each scale tensor has its K dimension reduced by a factor of `quantBlockSize`
    relative to the corresponding data tensor (i.e., kScale = ceil(K / quantBlockSize)).
    The scale type should be f8E8M0FNU (E8M0), encoding a power-of-two
    exponent per block: value = 2^(byte - 127).

    During RockToTTIR lowering, this op maps to `tt.dot_scaled` when scales are
    present, where the Triton hardware instruction (e.g.,
    V_MFMA_SCALE_F32_*_F8F6F4 on gfx950) consumes F8/F6/F4 data with E8M0
    scales natively.

    We might lower to `tt.dot_scaled` even if there are no scales as well, when
    `matrixAOrigElemType` or `matrixBOrigElemType` are defined. This is to avoid
    using `tt.dot` if we changed the datatype of A or B, it would be represented
    as i8 but it's really `matrixAOrigElemType`/`matrixBOrigElemType`.
  }];
  let assemblyFormat = [{
    `(` $matrixA  (`scaled` `by` $matrixScaleA^)? `,`
        $matrixB (`scaled` `by` $matrixScaleB^)? `,`
        $matrixC `)` attr-dict
    `:` type($matrixA) (`scaled` `by` type($matrixScaleA)^)? `,`
        type($matrixB) (`scaled` `by` type($matrixScaleB)^)? `,`
        type($matrixC)
    `->` type($result)
  }];
  let hasVerifier = 1;
}

// blockwise_load
def Rock_BlockwiseLoadOp
    : Rock_Op<"blockwise_load",
              [AllElementTypesMatch<["source", "result"]>]>,
      Arguments<(ins Arg<TensorOf<NativeMemoryOpTypes>,
                         "source global memory">:$source,
          Variadic<I32>:$sourceIndices)>,
      Results<(outs AnyRankedTensor:$result)> {
  let summary = "Blockwise load from global memory";
  let description = [{
    The `rock.blockwise_load` op loads at workgroup (block) level.
    We will load data from global memory.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source (`[` $sourceIndices^ `]`)? attr-dict
    `:` type($source) `->` type($result)
  }];
}

// blockwise_load_ptr
def Rock_BlockwiseLoadPtrOp
    : Rock_Op<"blockwise_load_ptr",
              [AllShapesMatch<["pointerTensor", "maskTensor", "result"]>]>,
      Arguments<(ins Arg<RankedTensorOf<[I32]>, "Pointer tensor">:$pointerTensor,
          Arg<RankedTensorOf<[I1]>, "Mask tensor">:$maskTensor)>,
      Results<(outs TensorOf<NativeMemoryOpTypes>:$result)> {
  let summary = "Blockwise load from global memory";
  let description = [{
  }];
  let assemblyFormat = [{
    $pointerTensor `[` $maskTensor `]` attr-dict
    `:` type($pointerTensor) `,` type($maskTensor) `->` type($result)
  }];
}

// blockwise_store
def Rock_BlockwiseStoreOp
    : Rock_Op<"blockwise_store",
              [DeclareOpInterfaceMethods<RockAcceptingViewOpInterface>,
               DeclareOpInterfaceMethods<RockWriterOpInterface>,
               AllElementTypesMatch<["source", "dest", "result"]>]>,
      Arguments<(ins Arg<TensorOf<NativeMemoryOpTypes>, "source">:$source,
          Arg<TensorOf<NativeMemoryOpTypes>, "destination view">:$dest,
          Variadic<I32>:$extraIndices, StoreMethodAttr:$storeMethod)>,
      Results<(outs AnyRankedTensor:$result)> {
  let summary = "Write out values to transformed destination";

  let description = [{
  }];

  let assemblyFormat = [{
    attr-dict $source  `->` $dest (`[` $extraIndices^ `]`)? `by` $storeMethod
    `:` type($source) `->` type($dest) `->` type($result)
  }];
  let hasVerifier = 1;
}

// blockwise_store_ptr
def Rock_BlockwiseStorePtrOp
    : Rock_Op<"blockwise_store_ptr",
              [AllShapesMatch<["pointerTensor", "maskTensor", "source"]>,
               AllElementTypesMatch<["source", "result"]>]>,
      Arguments<(ins Arg<RankedTensorOf<[I32]>, "Pointer tensor">:$pointerTensor,
          Arg<RankedTensorOf<[I1]>, "Mask tensor">:$maskTensor,
          Arg<TensorOf<NativeMemoryOpTypes>, "source">:$source,
          StoreMethodAttr:$storeMethod)>,
      Results<(outs TensorOf<NativeMemoryOpTypes>:$result)> {
  let summary = "Write out values to transformed destination";

  let description = [{
  }];

  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict $source  `->` $pointerTensor `(` $maskTensor `)` `by` $storeMethod
    `:` type($source) `->` type($pointerTensor) `(` type($maskTensor) `)` `->` type($result)
  }];
}

def Rock_TransformsToPtrOp
    : Rock_Op<"transforms_to_ptr",
              [AllShapesMatch<["pointers", "mask"]>]>,
      Arguments<(ins Arg<TensorOf<NativeMemoryOpTypes>,
                         "source">:$source,
          Variadic<I32>:$extraIndices)>,
      Results<(outs RankedTensorOf<[I32]>:$pointers,
               RankedTensorOf<[I1]>:$mask)> {
  let summary = "Convert transforms to pointer and mask tensors";

  let description = [{
    Converts a transformed tensor into pointer and mask tensors.
    The pointer tensor contains i32 offsets and the mask tensor contains i1 validity flags.
  }];

  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict $source (`[` $extraIndices^ `]`)?
    `:` type($source) `->` type($pointers) `,` type($mask)
  }];
}

// blockwise_reduction
def Rock_BlockwiseReduceOp
    : Rock_Op<"blockwise_reduce",
              [AllElementTypesMatch<["input", "result"]>]>,
      Arguments<(ins AnyRankedTensor:$input, IndexAttr:$axis,
          ReduceMethodAttr:$reduceMethod)>,
      Results<(outs AnyRankedTensor:$result)> {
  let summary = "Blockwise reduce operation";
  let description = [{
    The `rock.blockwise_reduce` op does reduce at a blockwise level and
    to fill all elements of the input tensor with reduced values. 
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    $reduceMethod $input attr-dict `:` type($input) `->` type($result)
  }];
}

def Rock_CastToPtrOp : Rock_Op<"cast_to_ptr",
                               [Pure, AllShapesMatch<["src", "result"]>]> {
  let summary = "Cast a tensor to a pointer tensor";

  let description = [{
    Casts a tensor to a pointer tensor representation.
  }];

  let arguments = (ins RankedTensorOf<[I32]>:$src);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def Rock_ExtractPtrOp : Rock_Op<"extract_ptr", [Pure]> {
  let summary = "Extract a pointer from a tensor as an i32";

  let description = [{
    The `rock.extract_ptr` op extracts the base pointer from a tensor and
    returns it as an i32. This operation is used to obtain the memory address
    of tensor data for pointer arithmetic computations.

    Please note that this op is only used as a glue between
    tensor inputs (rock IR) and pointer inputs (triton IR).
  }];

  let arguments = (ins AnyRankedTensor:$source);

  let results = (outs I32:$result);

  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // ROCK_OPS
