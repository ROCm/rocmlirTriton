//===- RockFoldOobBufferOps.cpp - erase provably OOB buffer stores --------===//
//
// Copyright 2026 AMD
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
//===----------------------------------------------------------------------===//
//
// Triton's BufferOpsEmitter predicates stores branchlessly: a masked-off lane
// gets an offset past the buffer descriptor's NumRecords and the hardware drops
// the access. That hides the predicate from dead code elimination, so for a
// GEMM whose M dimension is padded up to mPerBlock the whole padded-tile
// epilogue stays alive.
//
// rocMLIR removes the same stores while still in MLIR: `RockCleanMathPass`
// seeds an IntegerRangeAnalysis from the launch geometry, and once the
// predicate collapses the amdgpu.raw_buffer_store canonicalization sees a
// statically out-of-bounds index. This pass does the same thing one dialect
// level down, on the llvm/rocdl IR that the Triton lowering produces:
//
//   1. Seed the analysis by attaching `range` to the ROCDL id ops from the
//      kernel's `rock.block_size` / `rock.grid_size` attributes. ROCDL's
//      SpecialIdRegisterOps implement InferIntRangeInterface off that
//      attribute, mirroring NVVM's rangeable special registers.
//   2. Run the solver and erase every `rocdl.raw.ptr.buffer.store` whose
//      offset range lies entirely at or past the descriptor's NumRecords,
//      restricted to the descriptor configurations where the hardware really
//      discards the access.
//
// The dead epilogue arithmetic then dies to ordinary DCE, because everything
// feeding the erased stores is Pure.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "llvm/Support/Debug.h"

#include <limits>
#include <optional>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFOLDOOBBUFFEROPSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fold-oob-buffer-ops"

using namespace mlir;
using namespace mlir::dataflow;

namespace {

/// Buffer resource word-3 field positions, matching
/// `AMDGPU::RSRC_*` in llvm/lib/Target/AMDGPU/SIInstrInfo.h.
constexpr uint32_t kRsrcAddTidEnableBit = 1u << 23;
constexpr uint32_t kRsrcOobSelectShift = 28;
constexpr uint32_t kRsrcOobSelectMask = 0x3u;
/// OOB_SELECT == 3 ("raw buffer, structured with swizzle disabled") is the
/// configuration where GFX10+ hardware discards an access whose offset is at or
/// past NumRecords. Other encodings clamp or wrap instead.
constexpr uint32_t kRsrcOobSelectRaw = 3;

struct RockFoldOobBufferOpsPass
    : public rock::impl::RockFoldOobBufferOpsPassBase<
          RockFoldOobBufferOpsPass> {
  using RockFoldOobBufferOpsPassBase::RockFoldOobBufferOpsPassBase;
  void runOnOperation() override;
};

/// True when `arch` encodes buffer descriptors the way this pass decodes them.
///
/// GFX10 introduced the OOB_SELECT field that says whether an out-of-range
/// access is discarded, clamped, or wrapped; on GFX9 and earlier those bits
/// mean something else entirely. GFX12.5 widened NumRecords past 32 bits and
/// moved it out of the word this pass reads. So only the RDNA generations in
/// between can be decoded from the flags word alone.
bool hasRawBufferOobSelect(StringRef arch) {
  auto [chip, _] = rock::parseArchString(arch);
  switch (triton::amdgpu::TargetFeatures(chip).getISAFamily()) {
  case triton::amdgpu::ISAFamily::RDNA1:
  case triton::amdgpu::ISAFamily::RDNA2:
  case triton::amdgpu::ISAFamily::RDNA3:
  case triton::amdgpu::ISAFamily::GFX1170:
  case triton::amdgpu::ISAFamily::RDNA4:
    return true;
  case triton::amdgpu::ISAFamily::Unknown:
  case triton::amdgpu::ISAFamily::GCN5_1:
  case triton::amdgpu::ISAFamily::CDNA1:
  case triton::amdgpu::ISAFamily::CDNA2:
  case triton::amdgpu::ISAFamily::CDNA3:
  case triton::amdgpu::ISAFamily::CDNA4:
  case triton::amdgpu::ISAFamily::GFX1250:
    return false;
  }
  return false;
}

/// Return the constant integer `v` holds, if it is an `llvm.mlir.constant`.
std::optional<APInt> matchConstantInt(Value v) {
  auto cst = v.getDefiningOp<LLVM::ConstantOp>();
  if (!cst)
    return std::nullopt;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(cst.getValue()))
    return intAttr.getValue();
  return std::nullopt;
}

/// Attach `range` to the ROCDL workitem/workgroup id reads in `func` so that
/// ROCDL's InferIntRangeInterface implementation has a bound to publish.
///
/// Rock kernels are launched one-dimensionally, so `rock.block_size` bounds
/// `workitem.id.x` and `rock.grid_size` bounds `workgroup.id.x`. The other
/// dimensions are left alone and keep their default full range.
void seedIdRanges(LLVM::LLVMFuncOp func) {
  auto blockSizeAttr = func->getAttrOfType<IntegerAttr>("rock.block_size");
  auto gridSizeAttr = func->getAttrOfType<IntegerAttr>("rock.grid_size");

  auto annotate = [&](Operation *op, IntegerAttr sizeAttr) {
    if (!sizeAttr || op->hasAttr("range"))
      return;
    int64_t size = sizeAttr.getInt();
    if (size <= 0)
      return;
    auto resultType = dyn_cast<IntegerType>(op->getResult(0).getType());
    if (!resultType)
      return;
    unsigned width = resultType.getWidth();
    // LLVM's constant ranges are half-open, so the upper bound is the size
    // itself. Bail out on the degenerate case where that wraps.
    if (APInt(width, size).isZero())
      return;
    op->setAttr("range",
                LLVM::ConstantRangeAttr::get(op->getContext(), APInt(width, 0),
                                             APInt(width, size)));
  };

  func.walk([&](Operation *op) {
    if (isa<ROCDL::ThreadIdXOp>(op))
      annotate(op, blockSizeAttr);
    else if (isa<ROCDL::BlockIdXOp>(op))
      annotate(op, gridSizeAttr);
  });
}

/// Decode the descriptor `store` writes through, returning its NumRecords if
/// the descriptor is one where the hardware drops an out-of-bounds access.
std::optional<uint64_t>
getDiscardingNumRecords(ROCDL::RawPtrBufferStoreOp store) {
  auto rsrc = store.getRsrc().getDefiningOp<ROCDL::MakeBufferRsrcOp>();
  if (!rsrc)
    return std::nullopt;

  std::optional<APInt> stride = matchConstantInt(rsrc.getStride());
  std::optional<APInt> numRecords = matchConstantInt(rsrc.getNumRecords());
  std::optional<APInt> flags = matchConstantInt(rsrc.getFlags());
  if (!stride || !numRecords || !flags)
    return std::nullopt;

  // A non-zero stride makes NumRecords count structured records rather than
  // bytes, which changes what the offset is compared against.
  if (!stride->isZero())
    return std::nullopt;

  uint32_t flagBits = flags->getZExtValue();
  if (flagBits & kRsrcAddTidEnableBit)
    return std::nullopt;
  if (((flagBits >> kRsrcOobSelectShift) & kRsrcOobSelectMask) !=
      kRsrcOobSelectRaw)
    return std::nullopt;

  // Degenerate extents: a zero-length buffer would make every access foldable
  // on a technicality, and an all-ones extent is how "unbounded" is spelled.
  uint64_t extent = numRecords->getZExtValue();
  if (extent == 0 || numRecords->isAllOnes())
    return std::nullopt;
  return extent;
}

/// Return the inferred unsigned range of `v`, or nullopt when the solver has
/// nothing for it.
std::optional<ConstantIntRanges> getRange(DataFlowSolver &solver, Value v) {
  auto *lattice = solver.lookupState<IntegerValueRangeLattice>(v);
  if (!lattice || lattice->getValue().isUninitialized())
    return std::nullopt;
  return lattice->getValue().getValue();
}

/// True when every lane's byte offset provably lands at or past `numRecords`,
/// so the hardware discards the store.
bool isProvablyOutOfBounds(DataFlowSolver &solver,
                           ROCDL::RawPtrBufferStoreOp store,
                           uint64_t numRecords) {
  std::optional<ConstantIntRanges> offset = getRange(solver, store.getOffset());
  std::optional<ConstantIntRanges> soffset =
      getRange(solver, store.getSoffset());
  LLVM_DEBUG({
    llvm::dbgs() << "  store " << store << "\n    numRecords=" << numRecords
                 << " offset=";
    if (offset)
      llvm::dbgs() << *offset;
    else
      llvm::dbgs() << "<none>";
    llvm::dbgs() << " soffset=";
    if (soffset)
      llvm::dbgs() << *soffset;
    else
      llvm::dbgs() << "<none>";
    llvm::dbgs() << "\n";
  });
  if (!offset || !soffset)
    return false;

  // The address the hardware bounds-checks is offset + soffset, computed in 32
  // bits. Work in 64 bits so the sum cannot wrap under us, and require that it
  // cannot wrap in 32 bits either, since a wrapped sum could land back in
  // bounds.
  uint64_t minSum =
      offset->umin().getZExtValue() + soffset->umin().getZExtValue();
  uint64_t maxSum =
      offset->umax().getZExtValue() + soffset->umax().getZExtValue();
  if (maxSum > std::numeric_limits<uint32_t>::max())
    return false;
  return minSum >= numRecords;
}

void RockFoldOobBufferOpsPass::runOnOperation() {
  LLVM::LLVMFuncOp func = getOperation();
  if (func.isExternal())
    return;

  auto archAttr = func->getAttrOfType<StringAttr>("rock.arch");
  if (!archAttr || !hasRawBufferOobSelect(archAttr.getValue())) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-fold-oob-buffer-ops: skipping " << func.getName()
               << ", buffer descriptors are not decodable for arch '"
               << (archAttr ? archAttr.getValue() : "<none>") << "'\n");
    return;
  }

  seedIdRanges(func);

  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(func)))
    return signalPassFailure();

  DEBUG_WITH_TYPE("rock-fold-oob-buffer-ops-ranges", {
    func.walk([&](Operation *op) {
      for (Value res : op->getResults()) {
        if (!isa<IntegerType>(res.getType()))
          continue;
        llvm::dbgs() << op->getName() << " -> ";
        if (auto r = getRange(solver, res))
          llvm::dbgs() << *r;
        else
          llvm::dbgs() << "<uninitialized>";
        llvm::dbgs() << "  |  " << *op << "\n";
      }
    });
  });

  SmallVector<ROCDL::RawPtrBufferStoreOp> dead;
  func.walk([&](ROCDL::RawPtrBufferStoreOp store) {
    std::optional<uint64_t> numRecords = getDiscardingNumRecords(store);
    if (!numRecords) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  descriptor not discarding for " << store << "\n");
      return;
    }
    if (isProvablyOutOfBounds(solver, store, *numRecords))
      dead.push_back(store);
  });

  LLVM_DEBUG(llvm::dbgs() << "rock-fold-oob-buffer-ops: erasing " << dead.size()
                          << " out-of-bounds buffer stores in "
                          << func.getName() << "\n");

  for (ROCDL::RawPtrBufferStoreOp store : dead)
    store.erase();
}

} // end anonymous namespace
