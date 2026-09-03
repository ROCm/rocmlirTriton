//===- RockTuningParamAttrInterface.td - tuning params for the rock ops ---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright Advanced Micro Devices, Inc.
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

    The three parameter-wise methods (`getParamValues`, `getParamNames` and
    `cloneWithParamValues`) default to the members `Rock_PerfConfigSchema`
    generates from an attribute's field list, since a perf-config field and a
    tunable parameter are the same thing named twice. An attribute that carries
    such a schema therefore implements them by declaring its parameters and
    nothing else; one that does not has to write them itself.
  }];
  let cppNamespace = "::mlir::rock";

  let methods =
      [InterfaceMethod<
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
          Appends every parameter to `values`, in declaration order. Together
          with `cloneWithParamValues` this lets a tuner move values between
          configs without knowing what any parameter means, or even how many
          there are.
        }],
           /*retType=*/"void",
           /*methodName=*/"getParamValues",
           /*args=*/(ins "::llvm::SmallVectorImpl<int64_t>&":$values),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{
        $_attr.getPerfConfigValues(values);
      }]>,
       InterfaceMethod<
           /*desc=*/[{
          Returns a config of this attribute's kind whose parameters are
          `values`, or null if `values` is not the length `getParamValues`
          produces.
        }],
           /*retType=*/"::mlir::rock::RockTuningParamAttrInterface",
           /*methodName=*/"cloneWithParamValues",
           /*args=*/(ins "::llvm::ArrayRef<int64_t>":$values),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{
        if (values.size() != ConcreteAttr::getPerfConfigKeys().size())
          return nullptr;
        return ConcreteAttr::getFromPerfConfigValues($_attr.getContext(),
                                                     values);
      }]>,
       InterfaceMethod<
           /*desc=*/[{
          Appends the perf-config key of every parameter to `names`, in the same
          order as `getParamValues`. These are the keys of the `key=value` form,
          so a tuner can single out a parameter it has an opinion about without
          hardcoding where in the config it sits.
        }],
           /*retType=*/"void",
           /*methodName=*/"getParamNames",
           /*args=*/(ins "::llvm::SmallVectorImpl<::llvm::StringRef>&":$names),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{
        ::llvm::ArrayRef<::llvm::StringRef> keys =
            ConcreteAttr::getPerfConfigKeys();
        names.append(keys.begin(), keys.end());
      }]>,
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
          Gate for the `rock-set-reduction-layout` pass (warp
          redistribution onto the reduction dim). `-1` means "use the
          arch default" (the default, currently equivalent to off), `0`
          means off, `1` means on.
        }],
           /*retType=*/"int64_t",
           /*methodName=*/"getUseReductionLayout",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/"">,
       InterfaceMethod<
           /*desc=*/[{
          Gate for Triton's `OptimizeEpilogue` pass. `-1` selects the
          store-tail heuristic, `0` disables the pass, and `1` enables it
          unconditionally.
        }],
           /*retType=*/"int64_t",
           /*methodName=*/"getUseOptimizeEpilogue",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/"">,
       InterfaceMethod<
           /*desc=*/[{
          Gate for lowering an f32 `tt.dot` as three bf16 products instead of
          an IEEE f32 dot. `-1` follows the per-arch default, `0` forces the
          IEEE path, and `1` forces the decomposition.
        }],
           /*retType=*/"int64_t",
           /*methodName=*/"getUseBf16x3ForF32",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/"">,
  ];
}

#endif // ROCK_TUNING_PARAM_INTERFACE
