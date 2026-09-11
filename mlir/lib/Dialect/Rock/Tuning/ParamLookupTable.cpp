// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShardDb.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

// Both instantiations of ParamLookupTable share one table. That costs nothing
// in reachability: every candidate a lookup compares against carries the
// kernel type in its key, so a gemm key can only ever resolve to a gemm shard.
static const std::map<StringRef, const QuickTuningShard *> &getShardTable() {
  static const auto table = [] {
    std::map<StringRef, const QuickTuningShard *> shards;
    for (const QuickTuningShard &shard : getQuickTuningShards())
      shards.emplace(shard.key, &shard);
    return shards;
  }();
  return table;
}

template <typename ParamsType>
const std::map<StringRef, const QuickTuningShard *> &
ParamLookupTable<ParamsType>::getTable() {
  return getShardTable();
}

static size_t getQuickTuningListMax() {
  static const size_t listMax = [] {
    std::optional<std::string> env =
        llvm::sys::Process::GetEnv(kQuickTuningListMaxEnvVar);
    size_t parsed = 0;
    if (env && !StringRef(*env).getAsInteger(10, parsed) && parsed > 0)
      return parsed;
    return kQuickTuningListMaxDefault;
  }();
  return listMax;
}

// The set cover `shard` records, which is the list an unknown problem sweeps.
static SmallVector<StringRef> getCover(const QuickTuningShard &shard) {
  ArrayRef<const char *> configs = shard.getConfigs();
  SmallVector<StringRef> cover;
  cover.reserve(shard.numCover);
  for (uint16_t configIdx : shard.getCover())
    cover.push_back(configs[configIdx]);
  return cover;
}

// The quick-tuning list `shard` holds for `problemHash`.
static SmallVector<StringRef> narrowToProblem(const QuickTuningShard &shard,
                                              uint64_t problemHash) {
  // Nothing gates on how the key was reached, so a substituted key is probed
  // just like an exact one. A mismatched substitution cannot produce a false
  // hit: the kernel type is part of the hashed problem key, so a problem that
  // landed on another operation's shard simply misses.
  ArrayRef<uint64_t> problems = shard.getProblems();
  const uint64_t *hit = llvm::lower_bound(problems, problemHash);
  if (hit == problems.end() || *hit != problemHash)
    return getCover(shard);

  size_t problemIdx = hit - problems.begin();
  ArrayRef<const char *> configs = shard.getConfigs();
  SmallVector<StringRef> res;
  // The cap bounds the whole list, the recorded bests included, so that it
  // means what its name says even when set below the number of bests.
  // Non-split-K comes first, so the config that survives the tightest cap is
  // the one that is legal in every fusion context.
  size_t listMax = getQuickTuningListMax();
  // `front()` is what a skip-benchmarking consumer runs and what a perfdb miss
  // falls back to, so the non-split-K best has to lead.
  for (uint16_t configIdx :
       {shard.getBestNonSplitK(problemIdx), shard.getBestSplitK(problemIdx)}) {
    if (res.size() >= listMax)
      break;
    if (configIdx == kQuickTuningNoConfig)
      continue;
    StringRef config = configs[configIdx];
    if (!llvm::is_contained(res, config))
      res.push_back(config);
  }
  // Backfill so that a recorded best rejected by the fusion, or measured on a
  // problem that only resembles this one, still has the set cover behind it.
  for (StringRef config : getCover(shard)) {
    if (res.size() >= listMax)
      break;
    if (!llvm::is_contained(res, config))
      res.push_back(config);
  }
  return res;
}

template <typename ParamsType>
SmallVector<StringRef>
ParamLookupTable<ParamsType>::lookup(StringRef arch, KernelType op,
                                     Type dataType, uint64_t problemHash) {
  arch = normalizeArch(arch);
  auto key = makeKey(arch, op, dataType);
  LLVM_DEBUG(llvm::dbgs() << "Lookup for tuning parameters with key " << key
                          << "\n");

  static const auto &table = getTable();
  if (auto it = table.find(key); it != table.end())
    return narrowToProblem(*it->second, problemHash);

  StringRef fallbackKey = findFallback(key);
  if (fallbackKey.empty())
    llvm::report_fatal_error(Twine("Tuning parameters not found for key ") +
                             key);
  LLVM_DEBUG(llvm::dbgs() << "Falling back to tuning parameters with key "
                          << fallbackKey << "\n");
  return narrowToProblem(*table.at(fallbackKey), problemHash);
}

template <typename ParamsType>
StringRef
ParamLookupTable<ParamsType>::getFallbackDataType(StringRef dataType) {
  // Map datatypes without their own tuning entries to the closest datatype that
  // has them in a single hop. fp8 and i8 share the 8-bit MFMA/tile space; f4
  // has no 4-bit neighbour so it also borrows i8.
  return llvm::StringSwitch<StringRef>(dataType)
      .Case("fp8", "i8")
      .Case("f4", "i8")
      .Default(StringRef());
}

template <typename ParamsType>
StringRef
ParamLookupTable<ParamsType>::getFallbackKernelType(StringRef kernelType) {
  return llvm::StringSwitch<StringRef>(kernelType)
      .Case("gemmelementwisegemm", "attention")
      .Case("convelementwisegemm", "attention")
      .Default(StringRef());
}

template <typename ParamsType>
bool ParamLookupTable<ParamsType>::splitKey(StringRef key, StringRef &arch,
                                            StringRef &kernelType,
                                            StringRef &dataType) {
  // No component of a key contains the separator, so the first and last ones
  // delimit the kernel type. Requiring them to differ rejects malformed keys.
  auto firstSep = key.find(separator);
  auto lastSep = key.rfind(separator);
  if (firstSep == StringRef::npos || firstSep == lastSep)
    return false;
  arch = key.substr(0, firstSep);
  kernelType = key.substr(firstSep + 1, lastSep - firstSep - 1);
  dataType = key.substr(lastSep + 1);
  return true;
}

template <typename ParamsType>
StringRef ParamLookupTable<ParamsType>::pickClosestRelative(
    StringRef target, ArrayRef<StringRef> relatives) {
  auto it = std::lower_bound(relatives.begin(), relatives.end(), target);
  if (it == relatives.end())
    return relatives.back();
  if (it == relatives.begin())
    return relatives.front();

  auto prev = std::prev(it);
  // Bounded four-iterator form: a relative may be shorter than `target` (e.g.
  // gfx90a_gemm_f16 against gfx1100_gemm_f16), and the three-iterator form
  // would read past its end.
  auto mismatchNext =
      std::mismatch(target.begin(), target.end(), it->begin(), it->end());
  auto mismatchPrev =
      std::mismatch(target.begin(), target.end(), prev->begin(), prev->end());

  if (mismatchNext.first < mismatchPrev.first)
    return *prev;
  // If the mismatches are equal, prefer the larger (newer) candidate
  return *it;
}

template <typename ParamsType>
StringRef ParamLookupTable<ParamsType>::findFallback(StringRef target) {
  StringRef arch, kernelType, dataType;
  if (!splitKey(target, arch, kernelType, dataType))
    return StringRef();

  StringRef fallbackKernelType = getFallbackKernelType(kernelType);
  StringRef fallbackDataType = getFallbackDataType(dataType);

  static const auto &table = getTable();

  // The three axes we may substitute along are not equally cheap, so they are
  // nested cheapest-innermost.
  //
  // Data type is outermost, i.e. the last resort: tile shapes, kpack and the
  // instruction width are all tuned per precision. Architecture comes next,
  // because LDS capacity and the available MFMA/WMMA shapes differ across
  // chips. Swapping the kernel type is cheapest: the gemm+gemm and conv+gemm
  // fusions lower through the very same gridwise code as attention and share
  // its perf-config format, so an attention list tuned for *this* chip beats
  // the same fusion tuned for a different one.
  for (StringRef data : {dataType, fallbackDataType}) {
    if (data.empty())
      continue;
    for (bool crossArch : {false, true}) {
      for (StringRef kernel : {kernelType, fallbackKernelType}) {
        if (kernel.empty())
          continue;
        // Only `key` needs to outlive this iteration; every candidate compared
        // against it is a key owned by the table, and so is the result.
        std::string key =
            (Twine(arch) + Twine(separator) + kernel + Twine(separator) + data)
                .str();
        if (!crossArch) {
          if (auto it = table.find(key); it != table.end())
            return it->first;
          continue;
        }
        SmallVector<StringRef, 12> relatives = getRelatives(key);
        if (!relatives.empty())
          return pickClosestRelative(key, relatives);
      }
    }
  }

  return StringRef();
}

template <typename ParamsType>
SmallVector<StringRef, 12>
ParamLookupTable<ParamsType>::getRelatives(StringRef target) {
  // Fall back within the same gfx major family (gfxN...), e.g. gfx942 ->
  // gfx90a; never across families.
  constexpr auto fallbackArchPrefixLen = 4;
  auto suffixStart = target.find(separator);
  if (suffixStart == StringRef::npos)
    return {};
  // Everything but the architecture, e.g. "_gemm_f16". Comparing the two
  // suffixes directly keeps the kernel type and data type intact; deriving a
  // length from `target` and slicing `candidate` by it instead would underflow
  // whenever a candidate key is shorter than the suffix being matched, making
  // every same-family key look like a relative.
  const StringRef targetSuffix = target.substr(suffixStart);

  SmallVector<StringRef, 12> relatives;

  static const auto &table = getTable();
  for (const auto &entry : table) {
    StringRef candidate = entry.first;
    auto candidateSuffixStart = candidate.find(separator);
    if (candidateSuffixStart == StringRef::npos)
      continue;
    // If suffix and prefix match, then they are relatives
    if (candidate.substr(candidateSuffixStart) == targetSuffix &&
        target.starts_with(candidate.substr(0, fallbackArchPrefixLen))) {
      relatives.push_back(candidate);
    }
  }

  return relatives;
}

StringRef mlir::rock::normalizeArch(StringRef arch) {
  auto gfxPos = arch.find("gfx");
  if (gfxPos == StringRef::npos) {
    llvm_unreachable("Invalid architecture string");
  }
  auto remaining = arch.substr(gfxPos);
  auto endPos =
      remaining.find_if_not([](char c) { return llvm::isAlnum(c); }, 3);
  return remaining.substr(0, endPos);
}

std::string mlir::rock::getDataTypeString(Type dataType) {
  if (dataType.getIntOrFloatBitWidth() == 4 && isa<FloatType>(dataType)) {
    // We use a simplified "f4" for all 4-bit float types
    return "f4";
  } else if (dataType.getIntOrFloatBitWidth() == 8 &&
             isa<FloatType>(dataType)) {
    // There are several 8-bit float types, but we use "fp8" generically
    return "fp8";
  } else if (dataType.getIntOrFloatBitWidth() == 16 &&
             isa<FloatType>(dataType)) {
    // We use "f16" for bf16 and f16 generically
    return "f16";
  } else {
    std::string result;
    llvm::raw_string_ostream os(result);
    os << dataType;
    if (dataType.isInteger() && (result.at(0) == 's' || result.at(0) == 'u')) {
      // Integer types can be printed as "sint" or "uint"
      result.erase(result.begin());
    }
    return result;
  }
}

template <typename ParamsType>
std::string
ParamLookupTable<ParamsType>::getKernelTypeString(KernelType kernelType) {
  switch (kernelType) {
  case KernelType::ConvBwdData:
    // We use the same suffix for all convolution types
    return stringifyEnum(KernelType::Conv).lower();
  default:
    return stringifyEnum(kernelType).lower();
  }
}

template class mlir::rock::ParamLookupTable<GemmParamsAttr>;
template class mlir::rock::ParamLookupTable<GemmGemmParamsAttr>;
