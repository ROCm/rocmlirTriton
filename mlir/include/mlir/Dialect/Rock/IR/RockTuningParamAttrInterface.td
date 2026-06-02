//===- RockTuningParamAttrInterface.td - tuning params for the rock ops ---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockTuningParamAttrInterface, which abstracts definitions
// and methods of different tuning parameters for rock dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_TUNING_PARAM_INTERFACE
#define ROCK_TUNING_PARAM_INTERFACE

include "mlir/Dialect/Rock/IR/RockBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"


def RockTuningParamAttrInterface : AttrInterface<"RockTuningParamAttrInterface"> {
  let description = [{
    Common accessors for the tuning parameters of rock perf-config attributes.

    Both `Rock_GemmParamsAttr` (GEMM / convolution) and `Rock_GemmGemmParamsAttr`
    (gemm+gemm / attention) carry the same set of perf-config knobs; the only
    naming difference is that the gemm+gemm variant suffixes the first gemm's
    `m`/`n` tile with `G0`. To let generic code (validation, pretty-printing,
    heuristics) treat both attributes uniformly, the gemm+gemm attribute
    forwards `getMPerBlock`/`getNPerBlock` to its `G0` counterparts.
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [InterfaceMethod<
                     /*desc=*/[{
          Return serialize perfconfig (comma separated) string
        }],
                     /*retType=*/"void",
                     /*methodName=*/"getPerfConfigStr",
                     /*args=*/(ins "::llvm::SmallVectorImpl<char>&":$perfStr),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Number of values of m to process in each workgroup. For gemm+gemm
          this returns the first gemm's `mPerBlockG0`.
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getMPerBlock",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Number of values of n to process in each workgroup. For gemm+gemm
          this returns the first gemm's `nPerBlockG0`.
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getNPerBlock",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Number of values of k to process in each workgroup.
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getKPerBlock",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Number of matrix instructions per LDS load (see AccelerateAMDMatmul).
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getKpack",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Number of CTAs (cluster launch tiles).
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getNumCTAs",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Number of waves (warps) per workgroup. The workgroup size is
          `getWaveSize(arch) * getNumWaves()`.
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getNumWaves",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Size of the m/n dimension of the accelerator (XDL/WMMA) instruction.
          Zero means "let the heuristic choose".
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getMatrixInstrNonkdim",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Returns splitKFactor (the Split-k GEMM algorithm's split factor).
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getSplitKFactor",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Number of pipeline stages (matches Triton's `num_stages`).
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getNumStages",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Hint to the backend compiler for wavefronts per execution unit.
          Zero means "let the heuristic choose".
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getWavesPerEU",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Number of M tile blocks grouped together when mapping workgroups
          onto the grid; larger groups improve L2 locality across the
          grouped M-blocks at the cost of scheduling flexibility (see
          `makeGroupedGridLayout`). Zero means "let the heuristic choose".
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getGridGroupSize",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Overrides the per-arch default application for the async-copy
          schedule. -1 means "use the arch default".
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getUseAsyncCopy",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Overrides the per-arch default application for the block-pingpong
          schedule. -1 means "use the arch default".
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getUseBlockPingpong",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Overrides the per-arch default application for the
          in-thread-transpose pass. -1 means "use the arch default".
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getUseInThreadTranspose",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Overrides the convert-to-buffer-ops / canonicalize-pointers /
          optimize-buffer-op-ptr pass cluster. -1 means "use the arch
          default".
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getUseBufferOps",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Overrides buffer atomics in convert-to-buffer-ops (requires
          buffer ops to be enabled). -1 means "use the arch default".
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getUseBufferAtomics",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/[{
          Bitfield selector mirroring upstream Triton's
          `HIPOptions.schedule_hint`. -1 means "use the arch default", 0
          means "none".
        }],
                     /*retType=*/"int64_t",
                     /*methodName=*/"getScheduleHint",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
  ];
}

#endif // ROCK_TUNING_PARAM_INTERFACE
