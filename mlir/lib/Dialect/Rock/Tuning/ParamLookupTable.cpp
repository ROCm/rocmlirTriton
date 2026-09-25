// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/WithColor.h"

#include <cassert>
#include <cstdlib>

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

template <typename ParamsType>
SmallVector<StringRef> ParamLookupTable<ParamsType>::lookup(
    StringRef arch, KernelType op, Type dataType, bool supportsSplitK,
    std::optional<QuickTuningProblemKey> problemKey) {
  arch = normalizeArch(arch);
  auto key = makeKey(arch, op, dataType);
  LLVM_DEBUG(llvm::dbgs() << "Lookup for tuning parameters with key " << key
                          << "\n");

  // Set ROCMLIR_DISABLE_PER_PROBLEM_QUICK_TUNING to any value, including "0",
  // to fall back to the set cover everywhere. Presence-based like
  // ROCMLIR_DISABLE_LDS_BLACKLIST, and read once so getenv cannot race a
  // concurrent setenv.
  static const bool perProblemDisabled =
      std::getenv("ROCMLIR_DISABLE_PER_PROBLEM_QUICK_TUNING") != nullptr;

  // Deliberately no key fallback: a ranking only holds for the problem it was
  // measured on. Without a map for this key there is no ranking to miss, so
  // an unsupported problem mode is only worth diagnosing when one exists.
  if (problemKey && !perProblemDisabled) {
    const auto &problemMap = getProblemMap();
    if (auto it = problemMap.find(key); it != problemMap.end()) {
      if (!problemKey->unsupportedFields.empty()) {
        llvm::WithColor::warning(llvm::errs())
            << "per-problem quick tuning does not represent the current "
               "problem's "
            << problemKey->unsupportedFields
            << "; falling back to the set cover. Exhaustively tune this "
               "problem mode and regenerate the map before reusing "
               "per-problem results.\n";
      } else if (it->second.getKeyVersionHash() != problemKey->versionHash) {
        llvm::WithColor::warning(llvm::errs())
            << "ignoring per-problem quick-tuning map " << key
            << " keyed with table lookup key version hash "
            << it->second.getKeyVersionHash()
            << "; the current lookup key version hash is "
            << problemKey->versionHash
            << ". Regenerate the map with quickTuningGen.py.\n";
      } else {
        SmallVector<StringRef> perfConfigs =
            it->second.lookup(problemKey->hash);
        LLVM_DEBUG(llvm::dbgs() << "Per-problem lookup returned "
                                << perfConfigs.size() << " perfconfigs\n");
        if (!perfConfigs.empty())
          return perfConfigs;
      }
    }
  }

  const auto &table = supportsSplitK ? getTable() : getNoSplitKTable();
  const auto &pairedTable = supportsSplitK ? getNoSplitKTable() : getTable();
  auto it = table.find(key);
  if (it != table.end())
    return SmallVector<StringRef>(it->second);

  auto fallbackKey = findFallback(key);
  if (!fallbackKey.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Falling back to tuning parameters with key "
                            << fallbackKey << "\n");
    if (auto fallback = table.find(fallbackKey); fallback != table.end())
      return SmallVector<StringRef>(fallback->second);
    return SmallVector<StringRef>(pairedTable.at(fallbackKey));
  }

  llvm::report_fatal_error(Twine("Tuning parameters not found for key ") + key);
}

template <typename ParamsType>
StringRef
ParamLookupTable<ParamsType>::getFallbackDataType(StringRef dataType) {
  // Map datatypes without their own tuning entries to the closest datatype that
  // has them in a single hop. bf16 and f16 have the same element width, and so
  // the same tile footprint; fp8 and i8 share the 8-bit MFMA/tile space; f4
  // has no 4-bit neighbour so it also borrows i8.
  return llvm::StringSwitch<StringRef>(dataType)
      .Case("bf16", "f16")
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
  assert(!relatives.empty() &&
         "pickClosestRelative requires at least one relative");
  // getRelatives explicitly sorts the candidates merged from both lookup
  // tables so lower_bound can search them. Keep the assertion here to pin down
  // that producer/consumer contract.
  assert(llvm::is_sorted(relatives) && "relatives must be sorted ascending");

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
  //
  // Search both tables, starting with `target` before trying substitutions.
  for (StringRef data : {dataType, fallbackDataType}) {
    if (data.empty())
      continue;
    for (bool crossArch : {false, true}) {
      for (StringRef kernel : {kernelType, fallbackKernelType}) {
        if (kernel.empty())
          continue;
        // Only `key` needs to outlive this iteration; every candidate compared
        // against it is a key owned by a table, and so is the result.
        std::string key =
            (Twine(arch) + Twine(separator) + kernel + Twine(separator) + data)
                .str();
        if (!crossArch) {
          if (auto it = getTable().find(key); it != getTable().end())
            return it->first;
          if (auto it = getNoSplitKTable().find(key);
              it != getNoSplitKTable().end())
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

  auto appendRelatives = [&](const auto &table) {
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
  };
  appendRelatives(getTable());
  appendRelatives(getNoSplitKTable());

  // Each std::map is sorted independently, but concatenating their keys does
  // not preserve global ordering. Sorting also groups duplicate keys so the
  // following unique removes entries present in both tables.
  llvm::sort(relatives);
  relatives.erase(std::unique(relatives.begin(), relatives.end()),
                  relatives.end());
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
    // bf16 gets its own key rather than sharing f16's
    return isa<BFloat16Type>(dataType) ? "bf16" : "f16";
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

template <>
std::map<StringRef, ArrayRef<StringRef>>
ParamLookupTable<GemmParamsAttr>::buildTable() {
  return {
#define Gemm_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Gemm_LOOKUP_TABLE_GEN
  };
}

template <>
std::map<StringRef, ArrayRef<StringRef>>
ParamLookupTable<GemmParamsAttr>::buildNoSplitKTable() {
  return {
#define Gemm_NOSPLITK_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Gemm_NOSPLITK_LOOKUP_TABLE_GEN
  };
}

template <>
std::map<StringRef, ArrayRef<StringRef>>
ParamLookupTable<GemmGemmParamsAttr>::buildTable() {
  return {
#define GemmGemm_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_LOOKUP_TABLE_GEN
  };
}

template <>
std::map<StringRef, ArrayRef<StringRef>>
ParamLookupTable<GemmGemmParamsAttr>::buildNoSplitKTable() {
  return {
#define GemmGemm_NOSPLITK_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_NOSPLITK_LOOKUP_TABLE_GEN
  };
}

#define Gemm_PER_PROBLEM_DEFINITIONS_GEN
#include "QuickTuningProblemMap.inc"
#undef Gemm_PER_PROBLEM_DEFINITIONS_GEN

template <>
llvm::StringMap<QuickTuningProblemMap>
ParamLookupTable<GemmParamsAttr>::buildProblemMap() {
  return {
#define Gemm_PER_PROBLEM_LOOKUP_TABLE_GEN
#include "QuickTuningProblemMap.inc"
#undef Gemm_PER_PROBLEM_LOOKUP_TABLE_GEN
  };
}

#define GemmGemm_PER_PROBLEM_DEFINITIONS_GEN
#include "QuickTuningProblemMap.inc"
#undef GemmGemm_PER_PROBLEM_DEFINITIONS_GEN

template <>
llvm::StringMap<QuickTuningProblemMap>
ParamLookupTable<GemmGemmParamsAttr>::buildProblemMap() {
  return {
#define GemmGemm_PER_PROBLEM_LOOKUP_TABLE_GEN
#include "QuickTuningProblemMap.inc"
#undef GemmGemm_PER_PROBLEM_LOOKUP_TABLE_GEN
  };
}

template class mlir::rock::ParamLookupTable<GemmParamsAttr>;
template class mlir::rock::ParamLookupTable<GemmGemmParamsAttr>;
