// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-opt --rock-annotate-fusions %s | FileCheck %s

#collapse = affine_map<(d0) -> (0, d0 floordiv 8, d0 mod 8)>
#collapse_out = #rock.transform_map<#collapse by [
  <Merge{1, 8, 8} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>]
  bounds = [64] -> [1, 8, 8]>

#expand = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#expand_in = #rock.transform_map<#expand by [
  <Unmerge{8, 4} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>,
  <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 8, 4] -> [32]>

#collapse_lse = affine_map<(d0) -> (0, d0)>
#collapse_lse_out = #rock.transform_map<#collapse_lse by [
  <Merge{1, 8} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>]
  bounds = [8] -> [1, 8]>

// An unfused kernel comes out completely unannotated, so its key stays what it
// was before this pass existed and its tuning database rows remain reachable.

// CHECK-LABEL: func.func @unfused
// CHECK-NOT: rock.input_fusions
// CHECK-NOT: rock.output_fusions
func.func @unfused(%a: tensor<1x8x4xf32>, %b: tensor<1x4x8xf32>) -> tensor<1x8x8xf32>
    attributes {rock.kernel} {
  %0 = rock.gemm %a * %b : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  return %0 : tensor<1x8x8xf32>
}

// Every op on the way into an operand is an input fusion, on either operand and
// however long the chain.

// CHECK-LABEL: func.func @input_fusions
// CHECK-SAME: rock.input_fusions = ["mulf", "addf", "subf"]
// CHECK-NOT: rock.output_fusions
func.func @input_fusions(%a: tensor<1x8x4xf32>, %scale: tensor<1x8x4xf32>,
                         %bias: tensor<1x8x4xf32>, %b: tensor<1x4x8xf32>,
                         %shift: tensor<1x4x8xf32>) -> tensor<1x8x8xf32>
    attributes {rock.kernel} {
  %0 = arith.mulf %a, %scale : tensor<1x8x4xf32>
  %1 = arith.addf %0, %bias : tensor<1x8x4xf32>
  %2 = arith.subf %b, %shift : tensor<1x4x8xf32>
  %3 = rock.gemm %1 * %2 : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  return %3 : tensor<1x8x8xf32>
}

// The two sides land in their own attributes. The math dialect is named the
// same way as arith, so `erf` arrives whole rather than as a float-suffixed
// `er`.

// CHECK-LABEL: func.func @input_and_output_fusions
// CHECK-SAME: rock.input_fusions = ["mulf"]
// CHECK-SAME: rock.output_fusions = ["addf", "erf"]
func.func @input_and_output_fusions(%a: tensor<1x8x4xf32>, %scale: tensor<1x8x4xf32>,
                                    %b: tensor<1x4x8xf32>,
                                    %bias: tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
    attributes {rock.kernel} {
  %0 = arith.mulf %a, %scale : tensor<1x8x4xf32>
  %1 = rock.gemm %0 * %b : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  %2 = arith.addf %1, %bias : tensor<1x8x8xf32>
  %3 = math.erf %2 : tensor<1x8x8xf32>
  return %3 : tensor<1x8x8xf32>
}

// A view between the fused op and the kernel does not hide it.

// CHECK-LABEL: func.func @fusion_behind_a_view
// CHECK-SAME: rock.input_fusions = ["mulf"]
func.func @fusion_behind_a_view(%a: tensor<32xf32>, %scale: tensor<32xf32>,
                                %b: tensor<1x4x8xf32>) -> tensor<1x8x8xf32>
    attributes {rock.kernel} {
  %0 = arith.mulf %a, %scale : tensor<32xf32>
  %1 = rock.transform %0 by #expand_in : tensor<32xf32> to tensor<1x8x4xf32>
  %2 = rock.gemm %1 * %b : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  return %2 : tensor<1x8x8xf32>
}

// An epilogue reads operands of its own, and what computes them is part of the
// epilogue even though no walk from the kernel's result reaches it. Here the
// bias is itself an `addf` on a side branch.

// CHECK-LABEL: func.func @epilogue_side_branch
// CHECK-SAME: rock.output_fusions = ["addf", "addf"]
func.func @epilogue_side_branch(%a: tensor<1x8x4xf32>, %b: tensor<1x4x8xf32>,
                                %bias: tensor<1x8x8xf32>,
                                %extra: tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
    attributes {rock.kernel} {
  %0 = arith.addf %bias, %extra : tensor<1x8x8xf32>
  %1 = rock.gemm %a * %b : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  %2 = arith.addf %1, %0 : tensor<1x8x8xf32>
  return %2 : tensor<1x8x8xf32>
}

// A result consumed twice must not report the shared ops twice.

// CHECK-LABEL: func.func @shared_ops_counted_once
// CHECK-SAME: rock.output_fusions = ["addf", "mulf", "subf"]
func.func @shared_ops_counted_once(%a: tensor<1x8x4xf32>,
                                   %b: tensor<1x4x8xf32>) -> tensor<1x8x8xf32>
    attributes {rock.kernel} {
  %0 = rock.gemm %a * %b : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  %1 = arith.addf %0, %0 : tensor<1x8x8xf32>
  %2 = arith.mulf %0, %1 : tensor<1x8x8xf32>
  %3 = arith.subf %1, %2 : tensor<1x8x8xf32>
  return %3 : tensor<1x8x8xf32>
}

// `rock.transform` is a view rather than arithmetic, so it is never named. On
// the kernel's own operands that costs nothing the key does not already carry:
// a transpose became -transQ or -transO in the pass before this one, and the
// shapes are the problem's own dimensions.
//
// TODO: layout operations are not named yet. On a fused operand a view is
// handled below the key rather than in it, by `rock-narrow-redundant-loads`
// collapsing a broadcast load and by the vectorization analysis folding a
// `Broadcast` into the vector length. This is where to name one if a case turns
// up where a view decides which perf config wins.

// CHECK-LABEL: func.func @views_and_stores
// CHECK-NOT: rock.input_fusions
// CHECK-NOT: rock.output_fusions
func.func @views_and_stores(%a: tensor<1x8x4xf32>, %b: tensor<1x4x8xf32>,
                            %out: tensor<64xf32>) -> tensor<64xf32>
    attributes {rock.kernel} {
  %0 = rock.gemm %a * %b : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  %1 = rock.transform %0 by #collapse_out : tensor<1x8x8xf32> to tensor<64xf32>
  %2 = rock.store %1 to %out by set : tensor<64xf32> -> tensor<64xf32> to tensor<64xf32>
  return %2 : tensor<64xf32>
}

// A constant has no operands, so it is not something the kernel computes. The
// zero splat a relu compares against must not show up as a fusion.

// CHECK-LABEL: func.func @constants
// CHECK-SAME: rock.output_fusions = ["maxnumf"]
// CHECK-NOT: rock.input_fusions
func.func @constants(%a: tensor<1x8x4xf32>, %b: tensor<1x4x8xf32>) -> tensor<1x8x8xf32>
    attributes {rock.kernel} {
  %zero = arith.constant dense<0.000000e+00> : tensor<1x8x8xf32>
  %0 = rock.gemm %a * %b : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  %1 = arith.maxnumf %0, %zero : tensor<1x8x8xf32>
  return %1 : tensor<1x8x8xf32>
}

// An op reaching neither the kernel nor its result is some other computation in
// the same function. It shares a terminator with the kernel and nothing else.

// CHECK-LABEL: func.func @unrelated_work
// CHECK-NOT: rock.input_fusions
// CHECK-NOT: rock.output_fusions
func.func @unrelated_work(%a: tensor<1x8x4xf32>, %b: tensor<1x4x8xf32>,
                          %x: tensor<1x8x8xf32>)
    -> (tensor<1x8x8xf32>, tensor<1x8x8xf32>) attributes {rock.kernel} {
  %0 = rock.gemm %a * %b : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  %1 = arith.addf %x, %x : tensor<1x8x8xf32>
  return %0, %1 : tensor<1x8x8xf32>, tensor<1x8x8xf32>
}

// Without an op carrying the FusionRoot trait there is no problem to key on.

// CHECK-LABEL: func.func @no_fusion_root
// CHECK-NOT: rock.input_fusions
// CHECK-NOT: rock.output_fusions
func.func @no_fusion_root(%x: tensor<1x8x8xf32>, %y: tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
    attributes {rock.kernel} {
  %0 = arith.addf %x, %y : tensor<1x8x8xf32>
  %1 = arith.mulf %0, %y : tensor<1x8x8xf32>
  return %1 : tensor<1x8x8xf32>
}

// With two of them there is no single problem either, and the op between is
// fused onto neither in particular. Bail rather than guess.

// CHECK-LABEL: func.func @two_fusion_roots
// CHECK-NOT: rock.input_fusions
// CHECK-NOT: rock.output_fusions
func.func @two_fusion_roots(%a: tensor<1x8x4xf32>, %b: tensor<1x4x8xf32>,
                            %c: tensor<1x8x4xf32>) -> tensor<1x8x4xf32>
    attributes {rock.kernel} {
  %0 = rock.gemm %a * %b : tensor<1x8x4xf32> * tensor<1x4x8xf32> -> tensor<1x8x8xf32>
  %1 = arith.addf %0, %0 : tensor<1x8x8xf32>
  %2 = rock.gemm %1 * %c : tensor<1x8x8xf32> * tensor<1x8x4xf32> -> tensor<1x8x4xf32>
  return %2 : tensor<1x8x4xf32>
}

// A convolution is a fusion root just as a gemm is.

// CHECK-LABEL: func.func @conv_root
// CHECK-SAME: rock.input_fusions = ["mulf"]
// CHECK-SAME: rock.output_fusions = ["maxnumf"]
func.func @conv_root(%fil: tensor<4x1x2x1x1xf32>, %in: tensor<1x1x2x3x3xf32>,
                     %scale: tensor<1x1x2x3x3xf32>,
                     %lo: tensor<1x1x4x3x3xf32>) -> tensor<1x1x4x3x3xf32>
    attributes {rock.kernel} {
  %0 = arith.mulf %in, %scale : tensor<1x1x2x3x3xf32>
  %1 = rock.conv(%fil, %0) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["k", "g", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index]
  } : tensor<4x1x2x1x1xf32>, tensor<1x1x2x3x3xf32> -> tensor<1x1x4x3x3xf32>
  %2 = arith.maxnumf %1, %lo : tensor<1x1x4x3x3xf32>
  return %2 : tensor<1x1x4x3x3xf32>
}

// `rock.attention` is a single fusion root, so what its rewrite pattern absorbed
// is inside the op rather than around it. Neither the bias add in its
// `elementwise` region nor the log-sum-exp it returns is a fusion; they are
// attention itself, and reporting them would give a plain flash-attention
// kernel a different key from the same kernel without them. The scale on Q is a
// fusion, because it is arithmetic the kernel must do and it sits outside.

// CHECK-LABEL: func.func @attention_absorbs_its_own_work
// CHECK-SAME: rock.input_fusions = ["mulf"]
// CHECK-NOT: rock.output_fusions
func.func @attention_absorbs_its_own_work(%q: tensor<1x8x4xf32>, %scale: tensor<1x8x4xf32>,
                                          %k: tensor<1x4x8xf32>, %v: tensor<1x8x4xf32>,
                                          %bias: tensor<1x8x8xf32>)
    -> (tensor<1x8x4xf32>, tensor<8xf32>) attributes {rock.kernel} {
  %0 = arith.mulf %q, %scale : tensor<1x8x4xf32>
  %result, %lse = rock.attention{
    qk = %0 * %k : tensor<1x8x4xf32>, tensor<1x4x8xf32>
    qk = elementwise otherIns(%bias : tensor<1x8x8xf32>) {
    ^bb0(%acc: tensor<1x8x8xf32>, %b: tensor<1x8x8xf32>):
      %e = arith.addf %acc, %b : tensor<1x8x8xf32>
      rock.yield %e : tensor<1x8x8xf32>
    }
    softmax(qk) * %v : tensor<1x8x4xf32>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32,
     splitKV = 1 : i32} -> tensor<1x8x4xf32>, tensor<1x8xf32>
  %1 = rock.transform %lse by #collapse_lse_out : tensor<1x8xf32> to tensor<8xf32>
  return %result, %1 : tensor<1x8x4xf32>, tensor<8xf32>
}

// `rock.gemm_elementwise_gemm` covers both gemms, so the elementwise work
// between them is the op's own cost and is not reported. Work genuinely outside
// it still is, on both sides.
//
// TODO: should that inter-gemm work ever need to reach the key, it will have to
// be read out of the region deliberately.

// CHECK-LABEL: func.func @geg_absorbs_its_own_work
// CHECK-SAME: rock.input_fusions = ["mulf"]
// CHECK-SAME: rock.output_fusions = ["maxnumf"]
func.func @geg_absorbs_its_own_work(%a: tensor<1x8x4xf32>, %scale: tensor<1x8x4xf32>,
                                    %b: tensor<1x4x8xf32>, %c: tensor<1x8x4xf32>,
                                    %bias: tensor<1x8x8xf32>,
                                    %lo: tensor<1x8x4xf32>) -> tensor<1x8x4xf32>
    attributes {rock.kernel} {
  %0 = arith.mulf %a, %scale : tensor<1x8x4xf32>
  %1 = rock.gemm_elementwise_gemm{
    ab = %0 * %b : tensor<1x8x4xf32>, tensor<1x4x8xf32>
    ab = elementwise otherIns(%bias : tensor<1x8x8xf32>) {
    ^bb0(%acc: tensor<1x8x8xf32>, %bb: tensor<1x8x8xf32>):
      %e = arith.addf %acc, %bb : tensor<1x8x8xf32>
      rock.yield %e : tensor<1x8x8xf32>
    }
    out = ab * %c : tensor<1x8x4xf32>
  } -> tensor<1x8x4xf32>
  %2 = arith.maxnumf %1, %lo : tensor<1x8x4xf32>
  return %2 : tensor<1x8x4xf32>
}
