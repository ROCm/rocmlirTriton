//===- binding.cpp - pybind11 binding for AmdArchDb (test-only) -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure-plumbing pybind11 binding over the arch-string helpers in
// `mlir/Dialect/Rock/IR/AmdArchDb.h`. Per-arch semantics live there.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/DeviceInfo.h"

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>
#include <tuple>

namespace py = pybind11;

PYBIND11_MODULE(amd_arch_db, m) {
  m.doc() = "rocmlirTriton AmdArchDb bindings (in-tree test helper). "
            "See mlir/Dialect/Rock/IR/AmdArchDb.h for per-function semantics.";

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
      "prefer_bf16x3_for_f32_dot",
      [](const std::string &arch) {
        return mlir::rock::preferBf16x3ForF32Dot(arch);
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

  m.def(
      "infer_num_chiplets",
      [](const std::string &arch, int64_t numCUs) {
        return mlir::rock::inferNumChiplets(arch, numCUs);
      },
      py::arg("arch"), py::arg("num_cus"));

  m.def("get_native_device_count",
        []() { return mlir::rock::getNativeDeviceCount(); });

  m.def(
      "get_native_arch",
      [](unsigned deviceId) -> py::object {
        std::optional<mlir::rock::NativeDeviceInfo> info =
            mlir::rock::getNativeDeviceInfo(deviceId);
        if (!info)
          return py::none();
        return py::str(info->arch);
      },
      py::arg("device_id") = 0);

  m.def(
      "get_native_num_cu",
      [](const std::string &arch) -> py::object {
        std::optional<int64_t> numCUs = mlir::rock::getNativeNumCU(arch);
        if (!numCUs)
          return py::none();
        return py::int_(*numCUs);
      },
      py::arg("arch") = "");
}
