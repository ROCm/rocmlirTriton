//===- Pipelines.h - Rock pipelines ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_PIPELINES_H_
#define MLIR_DIALECT_ROCK_PIPELINES_H_

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

using namespace mlir::detail;
using namespace llvm::cl;

namespace mlir {
namespace rock {

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

//===--- Highlevel Pipeline
//------------------------------------------------===//
struct HighlevelOptions : public PassPipelineOptions<HighlevelOptions> {

  PassOptions::Option<bool> disableRock{
      *this, "disable-rock",
      desc("Disable Rock dialect targeting when bufferizing"), init(false)};
};

/// Adds the `highlevel` pipeline to the `OpPassManager`.
void buildHighlevelPipeline(OpPassManager &pm,
                            const HighlevelOptions &options = {});

//===--- Kernel Pipeline --------------------------------------------------===//
struct KernelOptions : public PassPipelineOptions<KernelOptions> {

  PassOptions::Option<std::string> arch{
      *this, "arch", desc("AMDGPU ISA version: e.g. gfx908"), init("")};
};

/// Adds the `kernel` pipeline to the `OpPassManager`.
void buildKernelPipeline(OpPassManager &pm, const KernelOptions &options = {});

//===--- Triton Pipeline --------------------------------------------------===//
struct TritonOptions : public PassPipelineOptions<TritonOptions> {

  PassOptions::Option<std::string> arch{
      *this, "arch", desc("AMDGPU ISA version: e.g. gfx908"), init("")};
  PassOptions::Option<int> numWarps{*this, "numWarps", desc("Number of warps"),
                                    init(4)};
  PassOptions::Option<int> numCTAs{*this, "numCTAs", desc("Number of CTAs"),
                                   init(1)};
  PassOptions::Option<int> numStages{*this, "numStages",
                                     desc("Number of stages"), init(2)};
  PassOptions::Option<int> matrixInstrNonkdim{
      *this, "matrixInstrNonkdim", desc("Matrix instruction non-k dimension"),
      init(16)};
  PassOptions::Option<int> kpack{*this, "kpack", desc("kpack"), init(1)};
};

/// Adds the `triton` pipeline to the `OpPassManager`.
void buildTritonPipeline(OpPassManager &pm, const TritonOptions &options = {});

//===--- Backend Pipeline -------------------------------------------------===//
struct BackendOptions : public PassPipelineOptions<BackendOptions> {

  PassOptions::Option<std::string> triple{
      *this, "triple", desc("AMDGPU target triple: amdgcn-amd-amdhsa"),
      init("")};
  PassOptions::Option<std::string> chip{
      *this, "chip", desc("AMDGPU ISA version: e.g. gfx908"), init("")};
  PassOptions::Option<std::string> features{
      *this, "features", desc("AMDGPU target features"), init("")};
  PassOptions::Option<int32_t> optLevel{
      *this, "opt-level", desc("GPU compiler optimization level"), init(3)};
  PassOptions::Option<bool> compile{
      *this, "compile", desc("should the serailization pass be run"),
      init(true)};
  PassOptions::Option<int> numWarps{*this, "numWarps", desc("Number of warps"),
                                    init(4)};
  PassOptions::Option<int> numCTAs{*this, "numCTAs", desc("Number of CTAs"),
                                   init(1)};
  PassOptions::Option<int> wavesPerEU{*this, "wavesPerEU",
                                      desc("Number of waves per EU"), init(0)};
  PassOptions::Option<bool> enableFpFusion{
      *this, "enableFpFusion", desc("Whether to enable FP fusion"), init(true)};
  PassOptions::Option<bool> allowFlushDenorm{
      *this, "allowFlushDenorm", desc("Whether to allow flush denorm"),
      init(false)};
  PassOptions::Option<bool> suppressDiagnostic{
      *this, "suppress-diagnostic",
      desc("should we suppress diagnostic messages"), init(false)};
  PassOptions::Option<std::string> dumpCpuSchedules{
      *this, "dump-cpu-schedules",
      desc("Path to dump CPU verifier IR and transform schedules"), init("")};
};

/// Adds the `backend` pipeline to the `OpPassManager`.
void buildBackendPipeline(OpPassManager &pm,
                          const BackendOptions &options = {});

/// Registers all pipelines for the `rock` dialect.
void registerPipelines();

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_PIPELINES_H_
