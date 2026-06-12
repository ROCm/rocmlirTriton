//===- binding.cpp - pybind11 binding for AmdArchDb (test-only) -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure-plumbing pybind11 binding over the arch-string + `Dtype` enum helpers
// in `mlir/Dialect/Rock/IR/AmdArchDb.h`. Per-arch semantics live there.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(amd_arch_db, m) {
  m.doc() = "rocmlirTriton AmdArchDb bindings (in-tree test helper). "
            "See mlir/Dialect/Rock/IR/AmdArchDb.h for per-function semantics.";

  py::enum_<mlir::rock::Dtype>(m, "Dtype")
      .value("F32", mlir::rock::Dtype::F32)
      .value("F16", mlir::rock::Dtype::F16)
      .value("BF16", mlir::rock::Dtype::BF16);

  m.def(
      "is_fast_atomic_add_supported",
      [](const std::string &arch, mlir::rock::Dtype dtype) {
        return mlir::rock::isFastAtomicAddSupported(arch, dtype);
      },
      py::arg("arch"), py::arg("dtype"));

  m.def(
      "is_fast_atomic_max_supported",
      [](const std::string &arch, mlir::rock::Dtype dtype) {
        return mlir::rock::isFastAtomicMaxSupported(arch, dtype);
      },
      py::arg("arch"), py::arg("dtype"));

  m.def(
      "arch_supports_accel_fp8",
      [](const std::string &arch) {
        return mlir::rock::archSupportsAccelFp8(arch);
      },
      py::arg("arch"));

  m.def(
      "arch_supports_scaled_gemm",
      [](const std::string &arch) {
        return mlir::rock::archSupportsScaledGemm(arch);
      },
      py::arg("arch"));

  m.def(
      "get_wave_size",
      [](const std::string &arch) { return mlir::rock::getWaveSize(arch); },
      py::arg("arch"));

  m.def(
      "get_vgprs_per_eu",
      [](const std::string &arch) { return mlir::rock::getVGPRsPerEU(arch); },
      py::arg("arch"));

  m.def(
      "get_max_kpack",
      [](const std::string &arch) { return mlir::rock::getMaxKpack(arch); },
      py::arg("arch"));
}
