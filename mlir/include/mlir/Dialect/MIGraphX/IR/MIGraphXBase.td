//===- MIGraphXBase.td - MIGraphX dialect definitions ---*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MIGRAPHX_BASE
#define MIGRAPHX_BASE

include "mlir/IR/OpBase.td"

def MIGraphX_Dialect : Dialect {
  let name = "migraphx";
  let cppNamespace = "mlir::migraphx";

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;

  // TODO(bump): the ops in this dialect print their inherent attributes (which
  // are property-backed) through `attr-dict`, mixed with the discardable ones.
  // Opting out of strict properties keeps that legacy spelling; migrating the
  // assembly formats (and every test) to `prop-dict` is a separate task.
  let useStrictPropertiesInAssemblyFormat = 0;
}

#endif // MIGRAPHX_BASE
