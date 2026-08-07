/**
   @file unittests/unittest_multipatch.cxx

   @brief Multi-patch container unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

namespace {
struct PatchValue {
  int value;
};
} // namespace

TEST(MultiPatch, AddsAccessesAndFindsSharedAndUniquePatches) {
  iganet::MultiPatch<PatchValue> patches;
  auto shared = std::make_shared<PatchValue>(PatchValue{10});
  EXPECT_EQ(patches.addPatch(shared), 0);
  EXPECT_EQ(patches.addPatch(std::make_unique<PatchValue>(PatchValue{20})), 1);
  EXPECT_EQ(patches.npatches(), 2);
  EXPECT_EQ(patches.patch(0).value, 10);
  EXPECT_EQ(patches.patch(1).value, 20);
  EXPECT_EQ(patches.findPatchIndex(*shared), 0);
  EXPECT_EQ(patches.findPatchIndex(&patches.patch(1)), 1);

  PatchValue absent{30};
  EXPECT_THROW((void)patches.findPatchIndex(absent), std::runtime_error);
}

TEST(MultiPatch, SupportsForwardReverseCopyMoveAndClear) {
  iganet::MultiPatch<PatchValue> patches;
  patches.addPatch(std::make_shared<PatchValue>(PatchValue{1}));
  patches.addPatch(std::make_shared<PatchValue>(PatchValue{2}));

  EXPECT_EQ((*patches.begin())->value, 1);
  EXPECT_EQ((*patches.rbegin())->value, 2);
  const auto &const_patches = patches;
  EXPECT_EQ(std::distance(const_patches.cbegin(), const_patches.cend()), 2);
  EXPECT_EQ(std::distance(const_patches.crbegin(), const_patches.crend()), 2);
  EXPECT_EQ(const_patches.patches().size(), 2);

  auto copied = patches;
  EXPECT_EQ(copied.npatches(), 2);
  auto moved = std::move(copied);
  EXPECT_EQ(moved.npatches(), 2);
  EXPECT_EQ(copied.npatches(), 0);
  moved.clear();
  EXPECT_EQ(moved.npatches(), 0);
}

#ifndef NDEBUG
TEST(MultiPatchDeathTest, RejectsOutOfRangePatchAccess) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  iganet::MultiPatch<PatchValue> patches;
  EXPECT_DEATH_IF_SUPPORTED((void)patches.patch(0), "");
}
#endif
