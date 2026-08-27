//===- FoldOobBufferOps.cpp - drop masked-off raw buffer accesses ---------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// Triton's AMD backend predicates buffer accesses branchlessly: a masked-off
// lane gets a `voffset` past the descriptor's `NumRecords`, which the hardware
// discards. That hides the predicate from dead code elimination, keeping the
// stored value and everything behind it alive.
//
// This pass drops the accesses whose predicate it proves false and whose
// out-of-bounds arm it proves past `numRecords`, both read out of the IR rather
// than assumed. `soffset` is matched structurally instead, since the hardware
// checks `voffset + soffset` and no per-value lattice relates two values.
//
// Two domains run because Triton's index math needs both: ranges for interval
// facts, known bits for the `xor`-based LinearLayout facts intervals widen
// away.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFOLDOOBBUFFEROPSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fold-oob-buffer-ops"

using namespace mlir;

namespace {

struct RockFoldOobBufferOpsPass
    : public rock::impl::RockFoldOobBufferOpsPassBase<
          RockFoldOobBufferOpsPass> {
  using RockFoldOobBufferOpsPassBase::RockFoldOobBufferOpsPassBase;
  void runOnOperation() override;
};

} // end namespace

// The dataflow types below cannot live in an anonymous namespace: MLIR derives
// the `TypeID` of the `dataflow::Lattice<...>` instantiation from the type
// name, which is not unique for anonymous types.
namespace mlir::rock::fold_oob {

/// Integer range inference extended to the LLVM and ROCDL dialects, neither of
/// which implements `InferIntRangeInterface`. Keeping the transfer functions
/// here, delegating to the same `intrange` helpers, avoids declaring the
/// interface on upstream ops.
struct LLVMIntegerRangeAnalysis : dataflow::IntegerRangeAnalysis {
  using dataflow::IntegerRangeAnalysis::IntegerRangeAnalysis;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMIntegerRangeAnalysis)

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
                 ArrayRef<dataflow::IntegerValueRangeLattice *> results) final;
};

/// Sparse lattice value holding `llvm::KnownBits`, so the transfer functions
/// can be LLVM's own. Empty is the lattice bottom.
class KnownBitsValue {
public:
  KnownBitsValue() = default;
  explicit KnownBitsValue(llvm::KnownBits known) : known(std::move(known)) {}

  /// Nothing known, which for a non-integer is a zero-width fact no query can
  /// read anything out of.
  static KnownBitsValue getUnknown(Type type) {
    auto intType = dyn_cast<IntegerType>(type);
    return KnownBitsValue(llvm::KnownBits(intType ? intType.getWidth() : 0));
  }

  bool isUninitialized() const { return !known.has_value(); }
  const llvm::KnownBits &getValue() const { return *known; }

  /// A union of value sets keeps only the bits both sides agree on. That can
  /// only drop bits, so the lattice converges without widening.
  static KnownBitsValue join(const KnownBitsValue &lhs,
                             const KnownBitsValue &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs.known->getBitWidth() != rhs.known->getBitWidth())
      return KnownBitsValue(llvm::KnownBits(lhs.known->getBitWidth()));
    return KnownBitsValue(lhs.known->intersectWith(*rhs.known));
  }

  bool operator==(const KnownBitsValue &rhs) const {
    if (isUninitialized() || rhs.isUninitialized())
      return isUninitialized() == rhs.isUninitialized();
    return known->Zero == rhs.known->Zero && known->One == rhs.known->One;
  }

  void print(raw_ostream &os) const {
    if (isUninitialized()) {
      os << "<uninitialized>";
      return;
    }
    os << "zero: " << known->Zero << ", one: " << known->One;
  }

private:
  std::optional<llvm::KnownBits> known;
};

using KnownBitsLattice = dataflow::Lattice<KnownBitsValue>;

/// Known-bits inference over the LLVM and ROCDL dialects, the second of the two
/// domains the fold reasons in. See the file comment for why one is not enough.
class KnownBitsAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<KnownBitsLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KnownBitsAnalysis)

  void setToEntryState(KnownBitsLattice *lattice) override {
    Value anchor = cast<Value>(lattice->getAnchor());
    propagateIfChanged(
        lattice, lattice->join(KnownBitsValue::getUnknown(anchor.getType())));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const KnownBitsLattice *> operands,
                               ArrayRef<KnownBitsLattice *> results) override;
};

} // namespace mlir::rock::fold_oob

using namespace mlir::rock::fold_oob;

//===----------------------------------------------------------------------===//
// Range inference
//===----------------------------------------------------------------------===//

static bool isScalarInt(Type type) { return isa<IntegerType>(type); }

/// The integer `value` holds, at its own width, if it is a constant.
///
/// `LLVM::ConstantOp` lets the attribute's width differ from the result's, so
/// the attribute as written is not the number the access sees. Translation to
/// LLVM IR resolves that the way `ModuleTranslation::getLLVMConstant` does, by
/// sign-extending a signless or signed attribute and zero-extending an unsigned
/// or `i1` one, and this follows it: reasoning about `-1 : i32` in an `i64` as
/// four billion rather than as all ones would put a live access out of bounds.
/// Every constant the pass reads comes through here for that reason.
static std::optional<APInt> matchConstantInt(Value value) {
  auto valueType = dyn_cast<IntegerType>(value.getType());
  IntegerAttr attr;
  if (!valueType || !matchPattern(value, m_Constant(&attr)))
    return std::nullopt;

  auto attrType = dyn_cast<IntegerType>(attr.getType());
  unsigned width = valueType.getWidth();
  if (attrType && (attrType.isUnsigned() || attrType.getWidth() == 1))
    return attr.getValue().zextOrTrunc(width);
  return attr.getValue().sextOrTrunc(width);
}

static intrange::OverflowFlags
translateOverflowFlags(LLVM::IntegerOverflowFlags flags) {
  intrange::OverflowFlags result = intrange::OverflowFlags::None;
  if (LLVM::bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nsw))
    result |= intrange::OverflowFlags::Nsw;
  if (LLVM::bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nuw))
    result |= intrange::OverflowFlags::Nuw;
  return result;
}

static std::optional<intrange::CmpPredicate>
translateCmpPredicate(LLVM::ICmpPredicate pred) {
  switch (pred) {
  case LLVM::ICmpPredicate::eq:
    return intrange::CmpPredicate::eq;
  case LLVM::ICmpPredicate::ne:
    return intrange::CmpPredicate::ne;
  case LLVM::ICmpPredicate::slt:
    return intrange::CmpPredicate::slt;
  case LLVM::ICmpPredicate::sle:
    return intrange::CmpPredicate::sle;
  case LLVM::ICmpPredicate::sgt:
    return intrange::CmpPredicate::sgt;
  case LLVM::ICmpPredicate::sge:
    return intrange::CmpPredicate::sge;
  case LLVM::ICmpPredicate::ult:
    return intrange::CmpPredicate::ult;
  case LLVM::ICmpPredicate::ule:
    return intrange::CmpPredicate::ule;
  case LLVM::ICmpPredicate::ugt:
    return intrange::CmpPredicate::ugt;
  case LLVM::ICmpPredicate::uge:
    return intrange::CmpPredicate::uge;
  }
  return std::nullopt;
}

static ConstantIntRanges constantRange(const APInt &value) {
  return ConstantIntRanges::constant(value);
}

/// Range of an `i1` holding a known truth value.
static ConstantIntRanges boolRange(bool value) {
  return constantRange(APInt(/*numBits=*/1, value));
}

/// Where warp specialization records the widened warp count. Triton has no
/// `Attr...Name` constant for this one, so every user upstream, both backends
/// included, spells the literal.
constexpr StringLiteral kTotalNumWarpsAttrName = "ttg.total-num-warps";

/// The value of a positive integer attribute on `op`, if it carries one.
static std::optional<int64_t> getPositiveInt(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr || attr.getValue().sle(0))
    return std::nullopt;
  return attr.getValue().getSExtValue();
}

/// Which of the launch's nested counts an id register runs over. The launch
/// nests them: lanes fill a wave, waves fill a workgroup, workgroups fill a
/// cluster, and clusters are what a Triton program is.
enum class IdSpace { Workitem, Wave, Workgroup, Cluster, ClusterWorkgroup };

/// The space `op` counts in, if it is an x-dimension ROCDL id register. Only x
/// is handled: for workitem ids that is a Triton invariant, and for the rest a
/// rock convention, since rock emits only ProgramIDDim::X and rock.grid_size is
/// one scalar. A convention is not a guarantee, so y and z decline.
static std::optional<IdSpace> getIdSpace(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<IdSpace>>(op)
      .Case<ROCDL::ThreadIdXOp>([](auto) { return IdSpace::Workitem; })
      // A wave id has no dimension to be the x of: it is the one register
      // Triton reads on the targets that have it, in place of dividing the
      // workitem id by the warp size.
      .Case<ROCDL::WaveId>([](auto) { return IdSpace::Wave; })
      .Case<ROCDL::BlockIdXOp>([](auto) { return IdSpace::Workgroup; })
      .Case<ROCDL::ClusterIdXOp>([](auto) { return IdSpace::Cluster; })
      .Case<ROCDL::ClusterWorkgroupIdXOp>(
          [](auto) { return IdSpace::ClusterWorkgroup; })
      .Default([](auto) { return std::nullopt; });
}

/// The launch bound on `op`'s result, if `op` is one of the ROCDL id registers.
/// A `range` attribute on the op wins, since the lowering may have narrowed it
/// further. Otherwise the bound is the count `rock::collectKernelInfo` derives
/// for the space the register runs over, since an id covers everything the
/// runtime is handed there.
static std::optional<ConstantIntRanges> inferIdRegisterRange(Operation *op) {
  std::optional<LLVM::ConstantRangeAttr> declared =
      llvm::TypeSwitch<Operation *, std::optional<LLVM::ConstantRangeAttr>>(op)
          .Case<ROCDL::ThreadIdXOp, ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp,
                ROCDL::BlockIdXOp, ROCDL::BlockIdYOp, ROCDL::BlockIdZOp,
                ROCDL::ClusterIdXOp, ROCDL::ClusterIdYOp, ROCDL::ClusterIdZOp,
                ROCDL::ClusterWorkgroupIdXOp, ROCDL::ClusterWorkgroupIdYOp,
                ROCDL::ClusterWorkgroupIdZOp, ROCDL::WaveId>(
              [](auto idOp) { return idOp.getRange(); })
          .Default([](auto) { return std::nullopt; });
  unsigned width = op->getNumResults() == 1
                       ? op->getResult(0).getType().getIntOrFloatBitWidth()
                       : 0;
  if (declared && width > 0) {
    // ROCDL carries the range as the half-open interval LLVM uses.
    APInt lower = declared->getLower().zextOrTrunc(width);
    APInt upper = declared->getUpper().zextOrTrunc(width);
    if (upper.ugt(lower))
      return ConstantIntRanges::fromUnsigned(lower, upper - 1);
  }

  std::optional<IdSpace> space = getIdSpace(op);
  auto module = op->getParentOfType<ModuleOp>();
  if (!space || !module)
    return std::nullopt;

  // The program count is the module-level per-kernel rock.grid_size, the one
  // `rock::collectKernelInfo` reads and hands the runtime. The kernel func
  // still carries the copy `rock-tensor-to-triton-ptr` took of it before the
  // TTIR lowering, but nothing reconciles the two afterwards, so reading the
  // func would risk a bound short of the launch. The workgroups each program
  // launches is Triton's cluster size. Every factor is read rather than
  // assumed, since a bound short of the launch erases work that runs.
  auto programCount = [&]() -> std::optional<int64_t> {
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func)
      return std::nullopt;
    return getPositiveInt(
        module, rock::GridSizeAttr::getModuleAttrName(func.getName()));
  };
  auto clusterSize = [&] {
    return getPositiveInt(module, triton::gpu::AttrNumCTAsName);
  };
  // Warp specialization widens the block and records the result in
  // ttg.total-num-warps, which is why the launch prefers it to ttg.num-warps.
  // Triton's own warp-specialization lowering agrees: it requires that
  // attribute and calls the wave id it reads there the absolute warp id.
  auto warpCount = [&]() -> std::optional<int64_t> {
    if (std::optional<int64_t> total =
            getPositiveInt(module, kTotalNumWarpsAttrName))
      return total;
    return getPositiveInt(module, triton::gpu::AttrNumWarpsName);
  };

  std::optional<int64_t> bound;
  switch (*space) {
  case IdSpace::Workitem: {
    std::optional<int64_t> warps = warpCount();
    std::optional<int64_t> lanes =
        getPositiveInt(module, triton::gpu::AttrNumThreadsPerWarp);
    if (!warps || !lanes)
      return std::nullopt;
    bound = llvm::checkedMul(*warps, *lanes);
    break;
  }
  case IdSpace::Wave:
    // A wave index inside the workgroup, so the lane count does not enter.
    bound = warpCount();
    break;
  case IdSpace::Workgroup: {
    // The flat grid, which is every cluster's workgroups.
    std::optional<int64_t> programs = programCount();
    std::optional<int64_t> cluster = clusterSize();
    if (!programs || !cluster)
      return std::nullopt;
    bound = llvm::checkedMul(*programs, *cluster);
    break;
  }
  case IdSpace::Cluster:
    // One cluster per program, so the cluster factor does not enter.
    bound = programCount();
    break;
  case IdSpace::ClusterWorkgroup:
    // A rank inside one cluster.
    bound = clusterSize();
    break;
  }
  if (!bound || width == 0)
    return std::nullopt;

  // A bound that does not fit the id's own width would narrow to something
  // tighter than the truth, so leave it to the entry state.
  if (!APInt(64, *bound - 1).isIntN(width))
    return std::nullopt;
  return ConstantIntRanges::fromUnsigned(APInt::getZero(width),
                                         APInt(width, *bound - 1));
}

/// Range of `op`'s single integer result given the ranges of its operands, or
/// nullopt when `op` is not one of the shapes handled here.
static std::optional<ConstantIntRanges>
inferLLVMResultRange(Operation *op, ArrayRef<ConstantIntRanges> args) {
  if (auto idRange = inferIdRegisterRange(op))
    return idRange;

  unsigned width = op->getResult(0).getType().getIntOrFloatBitWidth();

  if (isa<LLVM::ConstantOp>(op)) {
    if (std::optional<APInt> value = matchConstantInt(op->getResult(0)))
      return constantRange(*value);
    return std::nullopt;
  }
  if (isa<LLVM::ZeroOp>(op))
    return constantRange(APInt::getZero(width));

  // Casts and comparisons read their operand's width, everything else needs
  // all operands to be integers of the result's width.
  if (isa<LLVM::ZExtOp>(op))
    return intrange::extUIRange(args[0], width);
  if (isa<LLVM::SExtOp>(op))
    return intrange::extSIRange(args[0], width);
  if (isa<LLVM::TruncOp>(op))
    return intrange::truncRange(args[0], width);
  if (auto cmp = dyn_cast<LLVM::ICmpOp>(op)) {
    if (!isScalarInt(cmp.getLhs().getType()))
      return std::nullopt;
    std::optional<intrange::CmpPredicate> pred =
        translateCmpPredicate(cmp.getPredicate());
    if (!pred)
      return std::nullopt;
    if (std::optional<bool> known =
            intrange::evaluatePred(*pred, args[0], args[1]))
      return boolRange(*known);
    return ConstantIntRanges::fromUnsigned(APInt::getZero(1),
                                           APInt::getAllOnes(1));
  }
  if (auto select = dyn_cast<LLVM::SelectOp>(op)) {
    if (!isScalarInt(select.getCondition().getType()))
      return std::nullopt;
    const ConstantIntRanges &condition = args[0];
    if (condition.umin().isOne())
      return args[1];
    if (condition.umax().isZero())
      return args[2];
    return args[1].rangeUnion(args[2]);
  }
  // A lane read moves a value between lanes without changing it, so what held
  // across the wave still holds for the lane read out. The lane index is a
  // second operand of its own width, which is why this settles before the
  // equal-width check below rather than in the switch after it.
  if (isa<ROCDL::ReadfirstlaneOp, ROCDL::ReadlaneOp>(op))
    return args[0];

  if (llvm::any_of(op->getOperands(), [&](Value operand) {
        return !isScalarInt(operand.getType()) ||
               operand.getType().getIntOrFloatBitWidth() != width;
      }))
    return std::nullopt;

  return llvm::TypeSwitch<Operation *, std::optional<ConstantIntRanges>>(op)
      .Case<LLVM::AddOp>([&](auto addOp) {
        return intrange::inferAdd(
            args, translateOverflowFlags(addOp.getOverflowFlags()));
      })
      .Case<LLVM::SubOp>([&](auto subOp) {
        return intrange::inferSub(
            args, translateOverflowFlags(subOp.getOverflowFlags()));
      })
      .Case<LLVM::MulOp>([&](auto mulOp) {
        return intrange::inferMul(
            args, translateOverflowFlags(mulOp.getOverflowFlags()));
      })
      .Case<LLVM::ShlOp>([&](auto shlOp) {
        return intrange::inferShl(
            args, translateOverflowFlags(shlOp.getOverflowFlags()));
      })
      .Case<LLVM::UDivOp>([&](auto) { return intrange::inferDivU(args); })
      .Case<LLVM::SDivOp>([&](auto) { return intrange::inferDivS(args); })
      .Case<LLVM::URemOp>([&](auto) { return intrange::inferRemU(args); })
      .Case<LLVM::SRemOp>([&](auto) { return intrange::inferRemS(args); })
      .Case<LLVM::AndOp>([&](auto) { return intrange::inferAnd(args); })
      .Case<LLVM::OrOp>([&](auto orOp) {
        // Disjoint operands share no set bit, so the or is an add, and
        // inferAdd is tighter than the bitwise widening inferOr has to do.
        return orOp.getIsDisjoint() ? intrange::inferAdd(args)
                                    : intrange::inferOr(args);
      })
      .Case<LLVM::XOrOp>([&](auto) { return intrange::inferXor(args); })
      .Case<LLVM::LShrOp>([&](auto) { return intrange::inferShrU(args); })
      .Case<LLVM::AShrOp>([&](auto) { return intrange::inferShrS(args); })
      .Case<LLVM::UMinOp>([&](auto) { return intrange::inferMinU(args); })
      .Case<LLVM::UMaxOp>([&](auto) { return intrange::inferMaxU(args); })
      .Case<LLVM::SMinOp>([&](auto) { return intrange::inferMinS(args); })
      .Case<LLVM::SMaxOp>([&](auto) { return intrange::inferMaxS(args); })
      .Default([](auto) { return std::nullopt; });
}

LogicalResult LLVMIntegerRangeAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> results) {
  // Ops that carry the interface (arith, index, ...) are the base class's job.
  if (isa<InferIntRangeInterface>(op))
    return dataflow::IntegerRangeAnalysis::visitOperation(op, operands,
                                                          results);

  if (results.size() != 1 || !isScalarInt(op->getResult(0).getType())) {
    setAllToEntryStates(results);
    return success();
  }

  // Leave the result uninitialized until every operand has a range; the solver
  // revisits this operation once they do.
  SmallVector<ConstantIntRanges> args;
  args.reserve(operands.size());
  for (const dataflow::IntegerValueRangeLattice *lattice : operands) {
    if (lattice->getValue().isUninitialized())
      return success();
    args.push_back(lattice->getValue().getValue());
  }

  std::optional<ConstantIntRanges> inferred = inferLLVMResultRange(op, args);
  if (!inferred) {
    setAllToEntryStates(results);
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "Inferred " << *inferred << " for " << *op
                          << "\n");
  propagateIfChanged(results[0], results[0]->join(*inferred));
  return success();
}

//===----------------------------------------------------------------------===//
// Known-bits inference
//===----------------------------------------------------------------------===//

/// The bits a value in `range` must have clear, namely everything above the
/// most significant bit its maximum can reach.
static llvm::KnownBits knownBitsFromRange(const ConstantIntRanges &range) {
  unsigned width = range.umax().getBitWidth();
  llvm::KnownBits known(width);
  known.Zero = APInt::getBitsSetFrom(width, range.umax().getActiveBits());
  return known;
}

static std::optional<llvm::KnownBits>
evaluateKnownBitsCmp(LLVM::ICmpPredicate pred, const llvm::KnownBits &lhs,
                     const llvm::KnownBits &rhs) {
  std::optional<bool> known;
  switch (pred) {
  case LLVM::ICmpPredicate::eq:
    known = llvm::KnownBits::eq(lhs, rhs);
    break;
  case LLVM::ICmpPredicate::ne:
    known = llvm::KnownBits::ne(lhs, rhs);
    break;
  case LLVM::ICmpPredicate::slt:
    known = llvm::KnownBits::slt(lhs, rhs);
    break;
  case LLVM::ICmpPredicate::sle:
    known = llvm::KnownBits::sle(lhs, rhs);
    break;
  case LLVM::ICmpPredicate::sgt:
    known = llvm::KnownBits::sgt(lhs, rhs);
    break;
  case LLVM::ICmpPredicate::sge:
    known = llvm::KnownBits::sge(lhs, rhs);
    break;
  case LLVM::ICmpPredicate::ult:
    known = llvm::KnownBits::ult(lhs, rhs);
    break;
  case LLVM::ICmpPredicate::ule:
    known = llvm::KnownBits::ule(lhs, rhs);
    break;
  case LLVM::ICmpPredicate::ugt:
    known = llvm::KnownBits::ugt(lhs, rhs);
    break;
  case LLVM::ICmpPredicate::uge:
    known = llvm::KnownBits::uge(lhs, rhs);
    break;
  }
  if (!known)
    return llvm::KnownBits(/*BitWidth=*/1);
  return llvm::KnownBits::makeConstant(APInt(/*numBits=*/1, *known));
}

/// Known bits of `op`'s single integer result given those of its operands, or
/// nullopt when `op` is not one of the shapes handled here.
static std::optional<llvm::KnownBits>
inferKnownBits(Operation *op, ArrayRef<llvm::KnownBits> args) {
  if (std::optional<ConstantIntRanges> idRange = inferIdRegisterRange(op))
    return knownBitsFromRange(*idRange);

  unsigned width = op->getResult(0).getType().getIntOrFloatBitWidth();

  if (isa<LLVM::ConstantOp>(op)) {
    if (std::optional<APInt> value = matchConstantInt(op->getResult(0)))
      return llvm::KnownBits::makeConstant(*value);
    return std::nullopt;
  }
  if (isa<LLVM::ZeroOp>(op))
    return llvm::KnownBits::makeConstant(APInt::getZero(width));

  if (isa<LLVM::ZExtOp>(op))
    return args[0].zext(width);
  if (isa<LLVM::SExtOp>(op))
    return args[0].sext(width);
  if (isa<LLVM::TruncOp>(op))
    return args[0].trunc(width);
  if (auto cmp = dyn_cast<LLVM::ICmpOp>(op)) {
    if (!isScalarInt(cmp.getLhs().getType()))
      return std::nullopt;
    return evaluateKnownBitsCmp(cmp.getPredicate(), args[0], args[1]);
  }
  if (auto select = dyn_cast<LLVM::SelectOp>(op)) {
    if (!isScalarInt(select.getCondition().getType()))
      return std::nullopt;
    const llvm::KnownBits &condition = args[0];
    if (condition.isAllOnes())
      return args[1];
    if (condition.isZero())
      return args[2];
    return args[1].intersectWith(args[2]);
  }
  // Identical to the range side: the value crosses lanes unchanged, and the
  // lane index's width is not the result's.
  if (isa<ROCDL::ReadfirstlaneOp, ROCDL::ReadlaneOp>(op))
    return args[0];

  if (llvm::any_of(op->getOperands(), [&](Value operand) {
        return !isScalarInt(operand.getType()) ||
               operand.getType().getIntOrFloatBitWidth() != width;
      }))
    return std::nullopt;

  auto hasNsw = [](auto arithOp) {
    return LLVM::bitEnumContainsAny(arithOp.getOverflowFlags(),
                                    LLVM::IntegerOverflowFlags::nsw);
  };
  auto hasNuw = [](auto arithOp) {
    return LLVM::bitEnumContainsAny(arithOp.getOverflowFlags(),
                                    LLVM::IntegerOverflowFlags::nuw);
  };

  return llvm::TypeSwitch<Operation *, std::optional<llvm::KnownBits>>(op)
      .Case<LLVM::AddOp>([&](auto addOp) {
        return llvm::KnownBits::add(args[0], args[1], hasNsw(addOp),
                                    hasNuw(addOp));
      })
      .Case<LLVM::SubOp>([&](auto subOp) {
        return llvm::KnownBits::sub(args[0], args[1], hasNsw(subOp),
                                    hasNuw(subOp));
      })
      .Case<LLVM::MulOp>(
          [&](auto) { return llvm::KnownBits::mul(args[0], args[1]); })
      .Case<LLVM::ShlOp>([&](auto shlOp) {
        return llvm::KnownBits::shl(args[0], args[1], hasNuw(shlOp),
                                    hasNsw(shlOp));
      })
      .Case<LLVM::UDivOp>(
          [&](auto) { return llvm::KnownBits::udiv(args[0], args[1]); })
      .Case<LLVM::SDivOp>(
          [&](auto) { return llvm::KnownBits::sdiv(args[0], args[1]); })
      .Case<LLVM::URemOp>(
          [&](auto) { return llvm::KnownBits::urem(args[0], args[1]); })
      .Case<LLVM::SRemOp>(
          [&](auto) { return llvm::KnownBits::srem(args[0], args[1]); })
      .Case<LLVM::AndOp>([&](auto) { return args[0] & args[1]; })
      .Case<LLVM::OrOp>([&](auto) { return args[0] | args[1]; })
      .Case<LLVM::XOrOp>([&](auto) { return args[0] ^ args[1]; })
      .Case<LLVM::LShrOp>(
          [&](auto) { return llvm::KnownBits::lshr(args[0], args[1]); })
      .Case<LLVM::AShrOp>(
          [&](auto) { return llvm::KnownBits::ashr(args[0], args[1]); })
      .Case<LLVM::UMinOp>(
          [&](auto) { return llvm::KnownBits::umin(args[0], args[1]); })
      .Case<LLVM::UMaxOp>(
          [&](auto) { return llvm::KnownBits::umax(args[0], args[1]); })
      .Case<LLVM::SMinOp>(
          [&](auto) { return llvm::KnownBits::smin(args[0], args[1]); })
      .Case<LLVM::SMaxOp>(
          [&](auto) { return llvm::KnownBits::smax(args[0], args[1]); })
      .Default([](auto) { return std::nullopt; });
}

LogicalResult
KnownBitsAnalysis::visitOperation(Operation *op,
                                  ArrayRef<const KnownBitsLattice *> operands,
                                  ArrayRef<KnownBitsLattice *> results) {
  if (results.size() != 1 || !isScalarInt(op->getResult(0).getType())) {
    setAllToEntryStates(results);
    return success();
  }

  // Leave the result uninitialized until every operand has a fact; the solver
  // revisits this operation once they do.
  SmallVector<llvm::KnownBits> args;
  args.reserve(operands.size());
  for (const KnownBitsLattice *lattice : operands) {
    if (lattice->getValue().isUninitialized())
      return success();
    args.push_back(lattice->getValue().getValue());
  }

  std::optional<llvm::KnownBits> inferred = inferKnownBits(op, args);
  if (!inferred || inferred->hasConflict()) {
    setAllToEntryStates(results);
    return success();
  }

  propagateIfChanged(results[0],
                     results[0]->join(KnownBitsValue(std::move(*inferred))));
  return success();
}

//===----------------------------------------------------------------------===//
// The fold
//===----------------------------------------------------------------------===//

/// Matches an integer constant equal to `expected` at the value's own width, so
/// that e.g. `-1` matches the i32 constant `4294967295`.
static bool isIntConstant(Value value, int64_t expected) {
  std::optional<APInt> constant = matchConstantInt(value);
  return constant && *constant == APInt(64, static_cast<uint64_t>(expected))
                                      .zextOrTrunc(constant->getBitWidth());
}

/// `lhs >= rhs` as unsigned values at the wider of the two widths, since
/// offsets are 32-bit while a descriptor's `numRecords` is 64-bit.
static bool ugeAtCommonWidth(const APInt &lhs, const APInt &rhs) {
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  return lhs.zext(width).uge(rhs.zext(width));
}

/// Answers whether an `i1` is false on every execution, and what an integer's
/// unsigned bounds are, in either of the two abstract domains.
namespace {
class PredicateOracle {
public:
  LogicalResult run(Operation *scope) {
    ranges.load<dataflow::DeadCodeAnalysis>();
    ranges.load<LLVMIntegerRangeAnalysis>();
    knownBits.load<dataflow::DeadCodeAnalysis>();
    knownBits.load<KnownBitsAnalysis>();
    return success(succeeded(ranges.initializeAndRun(scope)) &&
                   succeeded(knownBits.initializeAndRun(scope)));
  }

  bool isProvablyFalse(Value condition) {
    std::optional<APInt> max = getUnsignedMax(condition);
    return max && max->isZero();
  }

  /// The tightest unsigned lower bound either domain proves.
  std::optional<APInt> getUnsignedMin(Value value) {
    return getBound(value, /*wantMin=*/true);
  }

  /// The tightest unsigned upper bound either domain proves.
  std::optional<APInt> getUnsignedMax(Value value) {
    return getBound(value, /*wantMin=*/false);
  }

private:
  /// Reads the bound each domain holds for `value` and keeps the tighter of the
  /// two, since both are sound. A value neither domain constrains gets its
  /// type's full range rather than nullopt, one loaded from memory included;
  /// nullopt is for a value no solver visited, as in unreachable code.
  std::optional<APInt> getBound(Value value, bool wantMin) {
    // Both domains hold a width-zero fact for a non-integer value, out of which
    // a bound reads as zero rather than as no bound at all. Every caller passes
    // an operand the op verifier types as an integer, so this only keeps that
    // from becoming a silent wrong answer if one ever does not.
    if (!isScalarInt(value.getType()))
      return std::nullopt;

    std::optional<APInt> result;
    auto keep = [&](const APInt &bound) {
      if (!result || (wantMin ? bound.ugt(*result) : bound.ult(*result)))
        result = bound;
    };
    if (const auto *range =
            ranges.lookupState<dataflow::IntegerValueRangeLattice>(value))
      if (!range->getValue().isUninitialized())
        keep(wantMin ? range->getValue().getValue().umin()
                     : range->getValue().getValue().umax());
    if (const auto *known = knownBits.lookupState<KnownBitsLattice>(value))
      if (!known->getValue().isUninitialized())
        keep(wantMin ? known->getValue().getValue().getMinValue()
                     : known->getValue().getValue().getMaxValue());
    return result;
  }

  DataFlowSolver ranges;
  DataFlowSolver knownBits;
};
} // end namespace

/// Whether a descriptor built with `flags` still has the hardware bounds-check
/// a raw access against `numRecords`. The flags operand lands in different
/// descriptor fields per target, so this follows the same split
/// `SITargetLowering::lowerPointerAsRsrcIntrin` does, exhaustively: a family
/// added in a Triton bump should fail to build rather than fall into a
/// `default`.
static bool flagsKeepBoundsCheck(triton::amdgpu::ISAFamily isaFamily,
                                 const APInt &flags) {
  uint64_t bits = flags.getZExtValue();
  // The flags operand becomes the descriptor's high dword. TID_ENABLE makes the
  // check wave-relative, which is not the bound reasoned about here.
  constexpr uint64_t tidEnable = UINT64_C(1) << 23;

  switch (isaFamily) {
  // gfx9 has no OOB_SELECT field, so a zero-stride buffer always checks the
  // offset against NumRecords.
  case triton::amdgpu::ISAFamily::GCN5_1:
  case triton::amdgpu::ISAFamily::CDNA1:
  case triton::amdgpu::ISAFamily::CDNA2:
  case triton::amdgpu::ISAFamily::CDNA3:
  case triton::amdgpu::ISAFamily::CDNA4:
    return !(bits & tidEnable);
  // Of the four OOB_SELECT encodings only 3 checks the offset against
  // NumRecords when the stride is zero; 2 disables bounds checking entirely.
  case triton::amdgpu::ISAFamily::RDNA1:
  case triton::amdgpu::ISAFamily::RDNA2:
  case triton::amdgpu::ISAFamily::RDNA3:
  case triton::amdgpu::ISAFamily::GFX1170:
  case triton::amdgpu::ISAFamily::RDNA4:
    return !(bits & tidEnable) && ((bits >> 28) & 3) == 3;
  // gfx1250 rebuilds the descriptor with a 45-bit NumRecords, shifting flags
  // left by 28 so only bits [3:0] survive: swizzle_enable, a one-bit OOB_select
  // that checks the offset either way at zero stride, and a two-bit type.
  case triton::amdgpu::ISAFamily::GFX1250: {
    constexpr uint64_t swizzleEnable = 1, typeMask = 3 << 2;
    return (bits & (swizzleEnable | typeMask)) == 0;
  }
  case triton::amdgpu::ISAFamily::Unknown:
    return false;
  }
  llvm_unreachable("unhandled ISAFamily in flagsKeepBoundsCheck");
}

/// The `numRecords` a masked lane's offset has to clear, or nullopt when `rsrc`
/// is not a descriptor whose bound this pass understands. A non-zero stride
/// turns the descriptor structured, changing both the addressing and the bounds
/// check, so only the raw form is accepted.
static std::optional<APInt>
getDescriptorBound(Value rsrc, triton::amdgpu::ISAFamily isaFamily,
                   PredicateOracle &oracle) {
  auto makeRsrc = rsrc.getDefiningOp<ROCDL::MakeBufferRsrcOp>();
  if (!makeRsrc || !isIntConstant(makeRsrc.getStride(), 0))
    return std::nullopt;
  std::optional<APInt> flags = matchConstantInt(makeRsrc.getFlags());
  if (!flags || !flagsKeepBoundsCheck(isaFamily, *flags))
    return std::nullopt;

  Value numRecords = makeRsrc.getNumRecords();
  std::optional<APInt> lowest = oracle.getUnsignedMin(numRecords);
  std::optional<APInt> highest = oracle.getUnsignedMax(numRecords);
  if (!lowest || !highest)
    return std::nullopt;

  // Zero and all ones are special in the descriptor rather than extents: they
  // mark a resource as unbound and as one whose size is not tracked. What an
  // access past either does varies by target, so a range that could be either
  // declines everywhere rather than modelling one target's rule. The field
  // width is what names all ones.
  unsigned fieldBits =
      isaFamily == triton::amdgpu::ISAFamily::GFX1250 ? 45 : 32;
  if (lowest->isZero() ||
      ugeAtCommonWidth(*highest, APInt::getMaxValue(fieldBits)))
    return std::nullopt;
  return highest;
}

/// Which of the emitter's two out-of-bounds encodings an access matched, or
/// none if the hardware is not guaranteed to discard it.
enum class OobShape { None, PlainSentinel, SplitSoffset };

static StringRef getShapeName(OobShape shape) {
  return shape == OobShape::SplitSoffset ? "split-soffset sentinel"
                                         : "plain sentinel";
}

/// Classifies the offset a masked-off lane presents. The hardware discards it
/// when `voffset + soffset` reaches `numRecords` without wrapping, since a
/// wrapped sum lands back at the start of the buffer.
///
/// A zero `soffset` leaves nothing to wrap, so the bound on `voffset` is
/// enough. Otherwise the two must be related, and the only relation the emitter
/// builds is the pair it creates when lifting a uniform part of the offset out:
///
///   %c      = llvm.icmp "sge" %x, 0
///   soffset = llvm.select %c, %x, 0
///   voffset = llvm.select %c, llvm.sub(-1, %x), %fallback
///
/// Under `%c` the sum is exactly all-ones rather than a wrap; otherwise
/// `soffset` is zero and `%fallback` is judged on its own.
static OobShape classifyOobOffset(Value voffset, Value soffset,
                                  const APInt &numRecords,
                                  PredicateOracle &oracle) {
  std::optional<APInt> soffsetMax = oracle.getUnsignedMax(soffset);
  if (soffsetMax && soffsetMax->isZero()) {
    std::optional<APInt> offsetMin = oracle.getUnsignedMin(voffset);
    if (offsetMin && ugeAtCommonWidth(*offsetMin, numRecords))
      return OobShape::PlainSentinel;
    return OobShape::None;
  }

  auto offsetSelect = voffset.getDefiningOp<LLVM::SelectOp>();
  auto soffsetSelect = soffset.getDefiningOp<LLVM::SelectOp>();
  if (!offsetSelect || !soffsetSelect ||
      offsetSelect.getCondition() != soffsetSelect.getCondition())
    return OobShape::None;

  Value lifted = soffsetSelect.getTrueValue();
  auto negated = offsetSelect.getTrueValue().getDefiningOp<LLVM::SubOp>();
  if (!negated || !isIntConstant(negated.getLhs(), -1) ||
      negated.getRhs() != lifted)
    return OobShape::None;

  // The shared condition bounds `%x`, and so bounds `-1 - %x` from below.
  // Without it a large `%x` would leave `voffset` inside the buffer.
  auto nonNegative = offsetSelect.getCondition().getDefiningOp<LLVM::ICmpOp>();
  if (!nonNegative || nonNegative.getPredicate() != LLVM::ICmpPredicate::sge ||
      nonNegative.getLhs() != lifted || !isIntConstant(nonNegative.getRhs(), 0))
    return OobShape::None;

  unsigned width = voffset.getType().getIntOrFloatBitWidth();
  if (!ugeAtCommonWidth(APInt::getSignedMinValue(width), numRecords))
    return OobShape::None;

  if (classifyOobOffset(offsetSelect.getFalseValue(),
                        soffsetSelect.getFalseValue(), numRecords,
                        oracle) == OobShape::None)
    return OobShape::None;
  return OobShape::SplitSoffset;
}

/// The descriptor, offsets and cache policy a buffer access presents to the
/// hardware, which the atomics spell three different ways.
struct BufferAccess {
  Value rsrc, voffset, soffset, aux;
};

/// Whether the cache policy `aux` could carry `CPol::VOLATILE`. That bit is not
/// a hardware one: LLVM turns it into `MOVolatile`, which has to reach the
/// hardware as written however the offset compares. A policy this cannot read
/// counts as volatile, since the alternative is dropping such an access.
static bool mayBeVolatile(Value aux) {
  constexpr unsigned volatileBit = 31;
  std::optional<APInt> bits = matchConstantInt(aux);
  if (!bits)
    return true;
  return bits->getBitWidth() > volatileBit && (*bits)[volatileBit];
}

/// The prefix every raw buffer atomic intrinsic shares.
constexpr StringLiteral kBufferAtomicIntrinsic =
    "llvm.amdgcn.raw.ptr.buffer.atomic.";

/// The access an atomic makes, if `op` is a buffer atomic whose result nobody
/// reads. A discarded atomic performs no read-modify-write, so with a dead
/// result it has nothing left to do. `RockToTTIR` drops the returned value on
/// every split-K and atomic-max output write.
static std::optional<BufferAccess> getUnreadAtomic(Operation *op) {
  // Unlike a discarded load's zero, what a discarded atomic returns is unknown.
  if (!op->use_empty())
    return std::nullopt;

  return llvm::TypeSwitch<Operation *, std::optional<BufferAccess>>(op)
      .Case<ROCDL::RawPtrBufferAtomicCmpSwap, ROCDL::RawPtrBufferAtomicFaddOp,
            ROCDL::RawPtrBufferAtomicFmaxOp, ROCDL::RawPtrBufferAtomicSmaxOp,
            ROCDL::RawPtrBufferAtomicUminOp>([](auto atomic) {
        return BufferAccess{atomic.getRsrc(), atomic.getOffset(),
                            atomic.getSoffset(), atomic.getAux()};
      })
      .Case<LLVM::CallIntrinsicOp>(
          [](LLVM::CallIntrinsicOp call) -> std::optional<BufferAccess> {
            if (!call.getIntrin().starts_with(kBufferAtomicIntrinsic))
              return std::nullopt;
            // However many data operands come first, the tail is always
            // (rsrc, voffset, soffset, aux). A miscount lands on something that
            // is not a descriptor, which `getDescriptorBound` declines.
            ValueRange args = call.getArgs();
            if (args.size() < 4)
              return std::nullopt;
            return BufferAccess{args[args.size() - 4], args[args.size() - 3],
                                args[args.size() - 2], args[args.size() - 1]};
          })
      .Default([](auto) { return std::nullopt; });
}

/// Classifies an access as droppable when its `voffset` is the out-of-bounds
/// arm of the emitter's predicating select, that predicate is statically false,
/// and the access is not volatile.
static OobShape classifyAccess(const BufferAccess &access,
                               triton::amdgpu::ISAFamily isaFamily,
                               PredicateOracle &oracle) {
  if (mayBeVolatile(access.aux))
    return OobShape::None;

  std::optional<APInt> numRecords =
      getDescriptorBound(access.rsrc, isaFamily, oracle);
  if (!numRecords)
    return OobShape::None;

  Value voffset = access.voffset;

  auto select = voffset.getDefiningOp<LLVM::SelectOp>();
  if (!select)
    return OobShape::None;

  OobShape shape = classifyOobOffset(select.getFalseValue(), access.soffset,
                                     *numRecords, oracle);
  if (shape == OobShape::None)
    return OobShape::None;

  return oracle.isProvablyFalse(select.getCondition()) ? shape : OobShape::None;
}

void RockFoldOobBufferOpsPass::runOnOperation() {
  LLVM::LLVMFuncOp func = getOperation();
  if (func.isExternal())
    return;

  // The descriptor's flags mean different things per target, so without an arch
  // there is no way to tell whether a masked offset is bounds-checked at all.
  FailureOr<StringAttr> arch = rock::getArchOnFunc(func);
  if (failed(arch))
    return;
  triton::amdgpu::ISAFamily isaFamily =
      std::get<0>(rock::getArch(arch->getValue()));

  PredicateOracle oracle;
  if (failed(oracle.run(func)))
    return signalPassFailure();

  SmallVector<std::pair<ROCDL::RawPtrBufferStoreOp, OobShape>> deadStores;
  SmallVector<std::pair<ROCDL::RawPtrBufferLoadOp, OobShape>> deadLoads;
  SmallVector<std::pair<Operation *, OobShape>> deadAtomics;
  func.walk([&](Operation *op) {
    if (auto store = dyn_cast<ROCDL::RawPtrBufferStoreOp>(op)) {
      OobShape shape = classifyAccess({store.getRsrc(), store.getOffset(),
                                       store.getSoffset(), store.getAux()},
                                      isaFamily, oracle);
      if (shape != OobShape::None)
        deadStores.emplace_back(store, shape);
    } else if (auto load = dyn_cast<ROCDL::RawPtrBufferLoadOp>(op)) {
      OobShape shape = classifyAccess(
          {load.getRsrc(), load.getOffset(), load.getSoffset(), load.getAux()},
          isaFamily, oracle);
      if (shape != OobShape::None)
        deadLoads.emplace_back(load, shape);
    } else if (std::optional<BufferAccess> atomic = getUnreadAtomic(op)) {
      OobShape shape = classifyAccess(*atomic, isaFamily, oracle);
      if (shape != OobShape::None)
        deadAtomics.emplace_back(op, shape);
    }
    // `rocdl.raw.ptr.buffer.load.async.lds` carries the same masked offsets but
    // is left alone: a discarded one leaves stale data in its LDS slot, so
    // neither erasing it nor zeroing it is a win.
  });

  for (auto [store, shape] : deadStores) {
    LLVM_DEBUG(llvm::dbgs() << "Erasing " << getShapeName(shape) << " store "
                            << *store << "\n");
    store.erase();
  }
  numErasedStores += deadStores.size();

  for (auto [atomic, shape] : deadAtomics) {
    LLVM_DEBUG(llvm::dbgs() << "Erasing " << getShapeName(shape) << " atomic "
                            << *atomic << "\n");
    atomic->erase();
  }
  numErasedAtomics += deadAtomics.size();

  OpBuilder builder(&getContext());
  for (auto [load, shape] : deadLoads) {
    LLVM_DEBUG(llvm::dbgs() << "Zeroing " << getShapeName(shape) << " load "
                            << *load << "\n");
    builder.setInsertionPoint(load);
    // A discarded raw buffer load reads as zero.
    Value zero =
        LLVM::ZeroOp::create(builder, load.getLoc(), load.getRes().getType());
    load.getRes().replaceAllUsesWith(zero);
    load.erase();
  }
  numFoldedLoads += deadLoads.size();
}
