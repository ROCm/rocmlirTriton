#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

template <typename ParamsType>
ArrayRef<StringRef> ParamLookupTable<ParamsType>::lookup(StringRef arch,
                                                         KernelType op,
                                                         Type dataType) {
  arch = normalizeArch(arch);
  auto key = makeKey(arch, op, dataType);
  LLVM_DEBUG(llvm::dbgs() << "Lookup for tuning parameters with key " << key
                          << "\n");

  static const auto &table = getTable();
  auto it = table.find(key);

  // A healthy exact match (more than one real config) wins outright.
  if (it != table.end() && 
      llvm::count_if(it->second, [](StringRef cfg) { return !cfg.empty(); }) > 1)
    return it->second;

  // Either the key is missing or its list is degenerate (<= 1 real config).
  // Look for a "close" relative; findFallback() prefers relatives with a useful
  // (>1 config) list so a single-config arch borrows a richer neighbour's list.
  auto fallbackKey = findFallback(key);
  if (!fallbackKey.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Falling back to tuning parameters with key "
                            << fallbackKey << "\n");
    return table.at(fallbackKey);
  }

  // No relative at all (e.g. an op/dtype family with a lone degenerate entry).
  // Prefer returning the degenerate exact entry over aborting.
  if (it != table.end())
    return it->second;

  llvm::report_fatal_error(Twine("Tuning parameters not found for key ") + key);
}

template <typename ParamsType>
StringRef ParamLookupTable<ParamsType>::findFallback(StringRef target) {
  auto relatives = getRelatives(target);
  if (relatives.empty())
    return StringRef();

  // Keep relatives whose own a valid list (> 1 real config); Only if every
  // relative is degenerate do we keep the full set (eg. fp8 on gfx900/gfx1000).
  static const auto &table = getTable();
  SmallVector<StringRef, 12> validRelatives;
  for (StringRef relative : relatives)
    if (llvm::count_if(table.at(relative), [](StringRef cfg) { return !cfg.empty(); }) > 1)
      validRelatives.push_back(relative);

  // Narrow to the useful relatives, then pick the closest among them below. If
  // every relative is degenerate we keep the full set so we still return one.
  if (!validRelatives.empty())
    relatives = std::move(validRelatives);

  for (const auto &relative : relatives) {
    llvm::errs() << "Relative: " << relative << "\n";
  }

  auto it = std::lower_bound(relatives.begin(), relatives.end(), target);
  if (it == relatives.end())
    return relatives.back();
  if (it == relatives.begin())
    return relatives.front();

  auto prev = std::prev(it);
  auto mismatchNext = std::mismatch(target.begin(), target.end(), it->begin());
  auto mismatchPrev =
      std::mismatch(target.begin(), target.end(), prev->begin());

  if (mismatchNext.first < mismatchPrev.first)
    return *prev;
  else
    // If the mismatches are equal, prefer the larger (newer) candidate
    return *it;
}

template <typename ParamsType>
SmallVector<StringRef, 12>
ParamLookupTable<ParamsType>::getRelatives(StringRef target) {
  // Fall back within the same gfx major family (gfxN...), e.g. gfx942 ->
  // gfx90a; never across families.
  constexpr auto fallbackArchPrefixLen = 4;
  const auto suffixLen = target.size() - target.find(separator);

  SmallVector<StringRef, 12> relatives;

  static const auto &table = getTable();
  for (const auto &entry : table) {
    StringRef candidate = entry.first;
    // If suffix and prefix match, then they are relatives
    if (target.ends_with(candidate.substr(candidate.size() - suffixLen)) &&
        target.starts_with(candidate.substr(0, fallbackArchPrefixLen))) {
      relatives.push_back(candidate);
    }
  }
  return relatives;
}

template <typename ParamsType>
StringRef ParamLookupTable<ParamsType>::normalizeArch(StringRef arch) {
  auto gfxPos = arch.find("gfx");
  if (gfxPos == StringRef::npos) {
    llvm_unreachable("Invalid architecture string");
  }
  auto remaining = arch.substr(gfxPos);
  auto endPos =
      remaining.find_if_not([](char c) { return llvm::isAlnum(c); }, 3);
  return remaining.substr(0, endPos);
}

template <typename ParamsType>
std::string
ParamLookupTable<ParamsType>::getKernelTypeString(KernelType kernelType) {
  switch (kernelType) {
  case KernelType::ConvBwdData:
  case KernelType::ConvBwdWeight:
    // We use the same suffix for all convolution types
    return stringifyEnum(KernelType::Conv).lower();
  default:
    return stringifyEnum(kernelType).lower();
  }
}

template <typename ParamsType>
std::string ParamLookupTable<ParamsType>::getDataTypeString(Type dataType) {
  if (dataType.getIntOrFloatBitWidth() == 4 &&
             isa<FloatType>(dataType)) {
    // We usa simplified "f4" for all 4-bit float types
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
ParamLookupTable<GemmGemmParamsAttr>::buildTable() {
  return {
#define GemmGemm_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_LOOKUP_TABLE_GEN
  };
}

template class mlir::rock::ParamLookupTable<GemmParamsAttr>;
template class mlir::rock::ParamLookupTable<GemmGemmParamsAttr>;
