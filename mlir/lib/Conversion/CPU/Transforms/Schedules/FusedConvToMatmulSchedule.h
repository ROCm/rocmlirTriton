//===- FusedConvToMatmulSchedule.h - Fused conv -> matmul shape -*- C++ -*-===//
//
// Copyright 2026 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This header declares the function that builds the "fused conv to matmul"
// transform schedule using the MLIR C++ API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_FUSEDCONVTOMATMULSCHEDULE_H
#define MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_FUSEDCONVTOMATMULSCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;

namespace cpu {

/// Build the "fused conv to matmul" transform module.
///
/// This schedule consumes the 8-D fused conv `linalg.generic` emitted by the
/// `--cpu-conv-to-gemm` pass (matched via the `rock.cpu_fused_conv` attr) and
/// drives it toward a matmul-shaped inner kernel that the existing tiling
/// and vectorization schedules can recognise.
///
/// The 8-D iteration space of the fused op is `(N, G, C, Ho, Wo, KC, Fh, Fw)`
/// with iter types `[par, par, par, par, par, red, red, red]`. We tile by 1
/// along `(N, Ho, Fh, Fw)` to peel them off as outer scalar loops, leaving an
/// inner op whose iter types are still `[par x 5, red x 3]` but whose extents
/// are unit on `(N, G, Ho, Fh, Fw)` and full on `(C, Wo, KC)`. We then fold
/// the unit extents away via slice-style reshapes so the inner op collapses
/// to a 3-D `(C, Wo, KC)` matmul shape with iter types `[par, par, red]`.
///
/// Downstream the existing `TilingSchedule` / `VectorizationSchedule` apply
/// to that matmul-shaped inner op without any changes.
///
/// This schedule is intentionally a no-op when no `rock.cpu_fused_conv` op is
/// present in the function (e.g. for plain matmul or fwd-conv verifiers that
/// `--cpu-conv-to-gemm` declined to rewrite). The transform interpreter
/// silently succeeds with an empty match in that case.
OwningOpRef<ModuleOp> buildFusedConvToMatmulSchedule(MLIRContext *ctx);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_FUSEDCONVTOMATMULSCHEDULE_H
