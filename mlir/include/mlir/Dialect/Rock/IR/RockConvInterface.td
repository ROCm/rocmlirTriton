//===- RockConvInterface.td - common ConvOps access methods ---------=== //
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//
//
// This file defines RockConvInterface, which groups common access methods
// conv-like operations
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_CONV_INTERFACE
#define ROCK_CONV_INTERFACE

include "mlir/IR/OpBase.td"

def RockConvInterface : OpInterface<"RockConvInterface"> {
  let description = [{
    Interface for convolution operations. Provides uniform access to the three
    tensors in the mathematical convolution equation: output = conv(filter, input).

    Each method returns the appropriate Value regardless of whether it is an
    operand or the computed result of the operation:
    - ConvOp:          filter=operand, input=operand,  output=result
    - ConvBwdDataOp:   filter=operand, input=result,   output=operand($gradient)
  }];
  let cppNamespace = "::mlir::rock";

  let methods =
      [InterfaceMethod<
           /*desc=*/
           [{Return the filter/weights tensor in convolution semantics.}],
           /*retType=*/"::mlir::TypedValue<::mlir::ShapedType>",
           /*methodName=*/"getConvFilter",
           /*args=*/(ins)>,
       InterfaceMethod<
           /*desc=*/
           [{Return the forward-input tensor in convolution semantics.}],
           /*retType=*/"::mlir::TypedValue<::mlir::ShapedType>",
           /*methodName=*/"getConvInput",
           /*args=*/(ins)>,
       InterfaceMethod<
           /*desc=*/
           [{Return the forward-output tensor in convolution semantics.}],
           /*retType=*/"::mlir::TypedValue<::mlir::ShapedType>",
           /*methodName=*/"getConvOutput",
           /*args=*/(ins)>,
       InterfaceMethod<
           /*desc=*/[{Return convolution padding values.}],
           /*retType=*/"ArrayAttr",
           /*methodName=*/"getPadding",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{ return $_op.getPadding(); }]>,
       InterfaceMethod<
           /*desc=*/[{Return convolution stride values.}],
           /*retType=*/"ArrayAttr",
           /*methodName=*/"getStrides",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{ return $_op.getStrides(); }]>,
       InterfaceMethod<
           /*desc=*/[{Return convolution dilation values.}],
           /*retType=*/"ArrayAttr",
           /*methodName=*/"getDilations",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{ return $_op.getDilations(); }]>,
  ];
}

#endif // ROCK_CONV_INTERFACE
