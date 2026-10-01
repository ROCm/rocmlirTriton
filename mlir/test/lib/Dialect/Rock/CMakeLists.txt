# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
add_rocmlir_test_library(MLIRRockTestPasses
  TestCollapseContiguousMerges.cpp
  TestVectorizationInference.cpp
  TestFunctionFusibility.cpp
  TestTransformationMapsUtils.cpp
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRRockOps
  MLIRRockTransforms
  MLIRRockUtility
  MLIRPass)
