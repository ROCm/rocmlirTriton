//===- RocmDeviceNameTests.cpp - Tests for RocmDeviceName -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/RocmDeviceName.h"
#include "gtest/gtest.h"

using namespace mlir;

// --- parse ---

TEST(RocmDeviceNameTest, ParseChipOnly) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("gfx90a")));
  EXPECT_EQ(dev.getChip(), "gfx90a");
  EXPECT_EQ(dev.getTriple(), "amdgcn-amd-amdhsa");
  EXPECT_TRUE(dev.getFeatures().empty());
}

TEST(RocmDeviceNameTest, ParseWithTriple) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("amdgcn-amd-amdhsa:gfx942")));
  EXPECT_EQ(dev.getChip(), "gfx942");
  EXPECT_EQ(dev.getTriple(), "amdgcn-amd-amdhsa");
  EXPECT_TRUE(dev.getFeatures().empty());
}

TEST(RocmDeviceNameTest, ParseWithFeatures) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("gfx90a:sramecc+:xnack-")));
  EXPECT_EQ(dev.getChip(), "gfx90a");
  auto &feats = dev.getFeatures();
  ASSERT_EQ(feats.size(), 2u);
  EXPECT_TRUE(feats.lookup("sramecc"));
  EXPECT_FALSE(feats.lookup("xnack"));
}

TEST(RocmDeviceNameTest, ParseTripleAndFeatures) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-")));
  EXPECT_EQ(dev.getTriple(), "amdgcn-amd-amdhsa");
  EXPECT_EQ(dev.getChip(), "gfx950");
  EXPECT_EQ(dev.getFeatures().size(), 2u);
}

// The MI450 A0 and B0 code paths are separated by target name: A0 builds as
// gfx1250-strict, B0 as gfx1250, and a strict code object is not loadable on
// B0. The "-strict" suffix is therefore part of the chip name and must survive
// parsing, including when a triple is present -- the triple is also '-'
// separated, so splitting it must not eat the suffix.
TEST(RocmDeviceNameTest, ParseStrictChip) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("gfx1250-strict")));
  EXPECT_EQ(dev.getChip(), "gfx1250-strict");
  EXPECT_EQ(dev.getTriple(), "amdgcn-amd-amdhsa");
  EXPECT_TRUE(dev.getFeatures().empty());
}

TEST(RocmDeviceNameTest, ParseStrictChipWithTripleAndFeatures) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("amdgcn-amd-amdhsa:gfx1250-strict:xnack-")));
  EXPECT_EQ(dev.getTriple(), "amdgcn-amd-amdhsa");
  EXPECT_EQ(dev.getChip(), "gfx1250-strict");
  ASSERT_EQ(dev.getFeatures().size(), 1u);
  EXPECT_FALSE(dev.getFeatures().lookup("xnack"));
}

TEST(RocmDeviceNameTest, ParseEmptyStringFails) {
  RocmDeviceName dev;
  EXPECT_TRUE(failed(dev.parse("")));
}

TEST(RocmDeviceNameTest, ParseNonGfxChipFails) {
  RocmDeviceName dev;
  EXPECT_TRUE(failed(dev.parse("non-gpu")));
}

TEST(RocmDeviceNameTest, ParseMalformedFeatureFails) {
  RocmDeviceName dev;
  EXPECT_TRUE(failed(dev.parse("gfx90a:sramecc")));
}

TEST(RocmDeviceNameTest, ParseFeatureOverride) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("gfx90a:xnack+:xnack-")));
  EXPECT_EQ(dev.getChip(), "gfx90a");
  EXPECT_FALSE(dev.getFeatures().lookup("xnack"));
}

// --- getFeaturesForBackend ---

TEST(RocmDeviceNameTest, GetFeaturesForBackendEmpty) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("gfx90a")));
  EXPECT_EQ(dev.getFeaturesForBackend(), "");
}

TEST(RocmDeviceNameTest, GetFeaturesForBackendSorted) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("gfx90a:xnack-:sramecc+")));
  EXPECT_EQ(dev.getFeaturesForBackend(), "+sramecc,-xnack");
}

TEST(RocmDeviceNameTest, GetFeaturesForBackendSingle) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("gfx90a:xnack+")));
  EXPECT_EQ(dev.getFeaturesForBackend(), "+xnack");
}

// --- getFullName ---

TEST(RocmDeviceNameTest, GetFullNameChipOnly) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("gfx90a")));
  llvm::SmallString<64> name;
  dev.getFullName(name);
  EXPECT_EQ(name, "amdgcn-amd-amdhsa:gfx90a");
}

TEST(RocmDeviceNameTest, GetFullNameWithFeatures) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("amdgcn-amd-amdhsa:gfx90a:xnack-:sramecc+")));
  llvm::SmallString<64> name;
  dev.getFullName(name);
  EXPECT_EQ(name, "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-");
}

TEST(RocmDeviceNameTest, GetFullNameRoundTrip) {
  RocmDeviceName dev;
  ASSERT_TRUE(succeeded(dev.parse("amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-")));
  llvm::SmallString<64> name;
  dev.getFullName(name);

  RocmDeviceName dev2;
  ASSERT_TRUE(succeeded(dev2.parse(name)));
  EXPECT_EQ(dev2.getChip(), dev.getChip());
  EXPECT_EQ(dev2.getTriple(), dev.getTriple());
  EXPECT_EQ(dev2.getFeatures().size(), dev.getFeatures().size());
  for (const auto &entry : dev.getFeatures()) {
    auto it = dev2.getFeatures().find(entry.getKey());
    ASSERT_NE(it, dev2.getFeatures().end());
    EXPECT_EQ(it->second, entry.getValue());
  }
}
