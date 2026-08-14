//===- RockGemmWrapperInterface.td - ops that wrap rock.gemm ---------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//
//
// This file defines RockGemmWrapperInterface, which abstracts convolutions and
// matrix multiplies to allow code to operate on them generically.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_GEMM_WRAPPER_INTERFACE
#define ROCK_GEMM_WRAPPER_INTERFACE

include "mlir/IR/OpBase.td"

def RockGemmWrapperInterface : OpInterface<"RockGemmWrapperInterface"> {
  let description = [{
    Interface to abstract away gemm-wrapping operators in the rock dialect,
    which mainly include convolutions that can be transformed to matrix
    multiplications.

    This should include functions to get common attributes.
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [InterfaceMethod<
                     /*desc=*/[{
          Return the KernelType of this op
        }],
                     /*retType=*/"::mlir::rock::KernelType",
                     /*methodName=*/"getKernelType",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Return the size of the matrix multiplication that this op will eventually
          perform.
        }],
                     /*retType=*/"::mlir::rock::GemmSize",
                     /*methodName=*/"getGemmSize",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Return the element type of [what will become] matrix A for this operation.
        }],
                     /*retType=*/"::mlir::Type",
                     /*methodName=*/"getAType",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Return the element type of [what will become] matrix B for this operation.
        }],
                     /*retType=*/"::mlir::Type",
                     /*methodName=*/"getBType",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Return the element type of [what will become] matrix C for this operation.
        }],
                     /*retType=*/"::mlir::Type",
                     /*methodName=*/"getCType",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Set the tuning parameters attribute of this operation

          This is needed for --affix-tuning-params to work and can go away if it does
        }],
                     /*retType=*/"void",
                     /*methodName=*/"setGemmParamsAttr",
                     /*args=*/(ins "::mlir::Attribute":$params),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/[{
          $_op->setAttr($_op.getParamsAttrName(), params);
        }]>,
                 InterfaceMethod<
                     /*desc=*/[{
          Get the tuning parameters attribute of this operation
        }],
                     /*retType=*/"std::optional<RockTuningParamAttrInterface>",
                     /*methodName=*/"getGemmParams",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/[{
          return $_op.getParams();
        }]>,
                 InterfaceMethod<
                     /*desc=*/[{
          Return the optional scale tensor for matrix A.
          Returns a null Value if no scale is present.
        }],
                     /*retType=*/"::mlir::Value",
                     /*methodName=*/"getScaleA",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/[{
          return ::mlir::Value();
        }]>,
                 InterfaceMethod<
                     /*desc=*/[{
          Return the optional scale tensor for matrix B.
          Returns a null Value if no scale is present.
        }],
                     /*retType=*/"::mlir::Value",
                     /*methodName=*/"getScaleB",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/[{
          return ::mlir::Value();
        }]>,
                 InterfaceMethod<
                     /*desc=*/[{
          Return the type of the optional scale tensor for matrix A.
          Returns a null Type if no scale is present.
        }],
                     /*retType=*/"::mlir::Type",
                     /*methodName=*/"getScaleAType",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/[{
          auto scale = $_op.getScaleA();
          return scale ? scale.getType() : ::mlir::Type();
        }]>,
                 InterfaceMethod<
                     /*desc=*/[{
          Return the type of the optional scale tensor for matrix B.
          Returns a null Type if no scale is present.
        }],
                     /*retType=*/"::mlir::Type",
                     /*methodName=*/"getScaleBType",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/[{
          auto scale = $_op.getScaleB();
          return scale ? scale.getType() : ::mlir::Type();
        }]>,
  ];
}

#endif // ROCK_GEM_WRAPPER_INTERFACE
