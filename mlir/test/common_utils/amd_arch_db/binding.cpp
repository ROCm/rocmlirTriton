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

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"

#include <pybind11/pybind11.h>

#include <string>
#include <tuple>

namespace py = pybind11;

PYBIND11_MODULE(amd_arch_db, m) {
  m.doc() = "rocmlirTriton AmdArchDb bindings (in-tree test helper). "
            "See mlir/Dialect/Rock/IR/AmdArchDb.h for per-function semantics.";

  py::enum_<mlir::rock::Dtype>(m, "Dtype")
      .value("F32", mlir::rock::Dtype::F32)
      .value("F16", mlir::rock::Dtype::F16)
      .value("BF16", mlir::rock::Dtype::BF16);

  using mlir::triton::amdgpu::ISAFamily;
  py::enum_<ISAFamily>(m, "ISAFamily")
      .value("Unknown", ISAFamily::Unknown)
      .value("GCN5_1", ISAFamily::GCN5_1)
      .value("CDNA1", ISAFamily::CDNA1)
      .value("CDNA2", ISAFamily::CDNA2)
      .value("CDNA3", ISAFamily::CDNA3)
      .value("CDNA4", ISAFamily::CDNA4)
      .value("RDNA1", ISAFamily::RDNA1)
      .value("RDNA2", ISAFamily::RDNA2)
      .value("RDNA3", ISAFamily::RDNA3)
      .value("GFX1170", ISAFamily::GFX1170)
      .value("RDNA4", ISAFamily::RDNA4)
      .value("GFX1250", ISAFamily::GFX1250);

  m.def(
      "get_isa_family",
      [](const std::string &arch) {
        return std::get<0>(mlir::rock::getArch(arch));
      },
      py::arg("arch"));

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
      "arch_supports_non_k_packed_scaled_input",
      [](const std::string &arch) {
        return mlir::rock::archSupportsNonKPackedScaledInput(arch);
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

  m.def(
      "supports_non_pow2_k_per_block",
      [](const std::string &arch) {
        return mlir::rock::supportsNonPow2KPerBlock(arch);
      },
      py::arg("arch"));
}
