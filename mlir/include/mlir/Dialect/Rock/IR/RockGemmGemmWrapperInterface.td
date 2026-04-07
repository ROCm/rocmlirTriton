//===- RockGemmGemmWrapperInterface.td - ops that wrap rock.attention
//---------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockGemmGemmWrapperInterface, which abstracts attention and
// gemm+gemm and friends (conv+gemm, ...) to allow code to operate on them
// generically.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_GEMM_GEMM_WRAPPER_INTERFACE
#define ROCK_GEMM_GEMM_WRAPPER_INTERFACE

include "mlir/IR/OpBase.td"

def RockGemmGemmWrapperInterface : OpInterface<"RockGemmGemmWrapperInterface"> {
  let description = [{
    Interface to abstract away gemm+gemm-wrapping operators in the rock dialect,
    which mainly include attention and gemm+gemm and friends that can be implemented
    with flash attention.

    This should include functions to get common attributes.
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [
    InterfaceMethod<
        /*desc=*/[{
          Return the KernelType of this op
        }],
        /*retType=*/"::mlir::rock::KernelType",
        /*methodName=*/"getKernelType",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Return the size of the matrix multiplication that this op will eventually
          perform.
        }],
        /*retType=*/"::mlir::rock::GemmGemmSize",
        /*methodName=*/"getGemmGemmSize",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Return the element type of [what will become] matrix A for this operation.
        }],
        /*retType=*/"::mlir::Type",
        /*methodName=*/"getAType",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the element type of [what will become] matrix B for this operation.
        }],
        /*retType=*/"::mlir::Type",
        /*methodName=*/"getBType",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the element type of [what will become] matrix C for this operation.
        }],
        /*retType=*/"::mlir::Type",
        /*methodName=*/"getCType",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the element type of [what will become] output matrix for this operation.
        }],
        /*retType=*/"::mlir::Type",
        /*methodName=*/"getOutType",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the whether matrix A is transposed.
        }],
        /*retType=*/"bool",
        /*methodName=*/"getTransposedA",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the whether matrix B is transposed.
        }],
        /*retType=*/"bool",
        /*methodName=*/"getTransposedB",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the whether matrix C is transposed.
        }],
        /*retType=*/"bool",
        /*methodName=*/"getTransposedC",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the whether output matrix is transposed.
        }],
        /*retType=*/"bool",
        /*methodName=*/"getTransposedOut",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Set the tuning parameters attribute of the first GEMM
        }],
        /*retType=*/"void",
        /*methodName=*/"setGemm0ParamsAttr",
        /*args=*/(ins "::mlir::Attribute":$params),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          $_op->setAttr($_op.getParams0AttrName(), params);
        }]
      >,
    InterfaceMethod<
        /*desc=*/[{
          Set the tuning parameters attribute of the second GEMM
        }],
        /*retType=*/"void",
        /*methodName=*/"setGemm1ParamsAttr",
        /*args=*/(ins "::mlir::Attribute":$params),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          $_op->setAttr($_op.getParams1AttrName(), params);
        }]
      >,
    InterfaceMethod<
        /*desc=*/[{
          Get the tuning parameters attribute of the first GEMM
        }],
        /*retType=*/"std::optional<RockTuningParamAttrInterface>",
        /*methodName=*/"getGemm0Params",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return $_op.getParams0();
        }]
      >,
    InterfaceMethod<
        /*desc=*/[{
          Get the tuning parameters attribute of the second GEMM
        }],
        /*retType=*/"std::optional<RockTuningParamAttrInterface>",
        /*methodName=*/"getGemm1Params",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return $_op.getParams1();
        }]
      >,
    InterfaceMethod<
        /*desc=*/[{
          Return the region corresponding to the fusion between first and second GEMM.
        }],
        /*retType=*/"Region&",
        /*methodName=*/"getPreSecondGemmRegion",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/ ""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Return a mutable range over the elementwise inputs for the fusion
          between first and second GEMM.
        }],
        /*retType=*/"::mlir::MutableOperandRange",
        /*methodName=*/"getPreSecondGemmElemwiseInputsMutable",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,

    // TODO: more methods here as needed
  ];
}

#endif // ROCK_GEMM_GEMM_WRAPPER_INTERFACE
