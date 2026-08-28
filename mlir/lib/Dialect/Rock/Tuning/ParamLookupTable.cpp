// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

template <typename ParamsType>
ArrayRef<StringRef> ParamLookupTable<ParamsType>::lookup(StringRef arch,
                                                         KernelType op,
                                                         Type dataType) {
  arch = normalizeArch(arch);
  // gfx1170 ships no tuned quick-tuning tables of its own. The generic
  // lexicographic fallback would land on gfx1151 (RDNA3.5 APU), whose top
  // conv f16 tile (128x256x32) does not fit the WMMA conv LDS budget on a
  // 64 KiB arch. Reuse gfx1200's (RDNA4) tables instead -- gfx1200 owns the
  // GEMM tables and cross-falls-back to gfx1201 for conv/attention, all of
  // which fit. This alias is scoped to tuning-table selection only; WMMA
  // detection, LDS sizing, and pipeline gating still use the real gfx1170.
  // TODO(AIROCMLIR-705): Generate a proper quick-tuning table for gfx1170 and
  // remove this alias.
  if (arch == "gfx1170")
    arch = "gfx1200";
  auto key = makeKey(arch, op, dataType);
  LLVM_DEBUG(llvm::dbgs() << "Lookup for tuning parameters with key " << key
                          << "\n");

  static const auto &table = getTable();
  auto it = table.find(key);
  if (it != table.end())
    return it->second;

  auto fallbackKey = findFallback(key);
  if (!fallbackKey.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Falling back to tuning parameters with key "
                            << fallbackKey << "\n");
    return table.at(fallbackKey);
  }

  llvm::report_fatal_error(Twine("Tuning parameters not found for key ") + key);
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
StringRef ParamLookupTable<ParamsType>::findFallback(StringRef target) {
  // `fallbackKey` owns the storage for the substituted-datatype key (if any).
  std::string fallbackKey;
  SmallVector<StringRef, 12> relatives = getRelatives(target);

  // When there is no same-datatype relative across architectures, borrow the
  // closest datatype that has tuning entries (e.g. gfx942_gemm_fp8 ->
  // gfx942_gemm_i8) and search once more. If that datatype has no relatives
  // either, give up.
  if (relatives.empty()) {
    auto dataTypePos = target.rfind(separator);
    if (dataTypePos == StringRef::npos)
      return StringRef();
    StringRef fallbackDataType =
        getFallbackDataType(target.substr(dataTypePos + 1));
    if (fallbackDataType.empty())
      return StringRef();
    fallbackKey =
        (Twine(target.substr(0, dataTypePos + 1)) + fallbackDataType).str();
    target = fallbackKey;
    relatives = getRelatives(target);
    if (relatives.empty())
      return StringRef();
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
