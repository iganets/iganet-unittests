/**
   @file unittests/unittest_multipatch.cxx

   @brief Multi-patch container unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <gtest/gtest.h>
#include <iganet.h>

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

TEST(MultiPatch, AddsAccessesIteratesAndRemovesInterfaces) {
  iganet::MultiPatch<PatchValue> patches;
  auto patch4 = std::make_shared<PatchValue>(PatchValue{4});
  auto patch6 = std::make_shared<PatchValue>(PatchValue{6});
  patches.addPatch(patch4);
  patches.addPatch(patch6);

  EXPECT_EQ(patches.addInterface(0, iganet::south, 1, iganet::north), 0);
  ASSERT_EQ(patches.ninterfaces(), 1);

  auto &interface = patches.interface(0);
  EXPECT_EQ(&interface.firstPatch(), patch4.get());
  EXPECT_EQ(interface.firstSide(), iganet::south);
  EXPECT_EQ(&interface.secondPatch(), patch6.get());
  EXPECT_EQ(interface.secondSide(), iganet::north);
  EXPECT_EQ(patches.findInterfaceIndex(interface), 0);
  EXPECT_EQ(patches.findInterfaceIndex(&interface), 0);

  iganet::PatchInterface<PatchValue> absent(patch4, iganet::south, patch6,
                                            iganet::north);
  EXPECT_THROW((void)patches.findInterfaceIndex(absent), std::runtime_error);

  std::size_t count = 0;
  for (const auto &patchInterface : patches.interfaces()) {
    EXPECT_EQ(patchInterface.patch(0).value, 4);
    EXPECT_EQ(patchInterface.side(1), iganet::north);
    ++count;
  }
  EXPECT_EQ(count, 1);

  auto copied = patches;
  ASSERT_EQ(copied.ninterfaces(), 1);
  EXPECT_EQ(&copied.interface(0).patch(0), patch4.get());

  patches.removeInterface(0);
  EXPECT_EQ(patches.ninterfaces(), 0);
  EXPECT_EQ(patches.npatches(), 2);
}

TEST(MultiPatch, RejectsInterfacesWithForeignPatchesOrInvalidSides) {
  iganet::MultiPatch<PatchValue> patches;
  auto local = std::make_shared<PatchValue>(PatchValue{1});
  auto foreign = std::make_shared<PatchValue>(PatchValue{2});
  patches.addPatch(local);

  EXPECT_THROW(patches.addInterface(local, iganet::east, foreign, iganet::west),
               std::invalid_argument);
  EXPECT_THROW(patches.addInterface(local, iganet::none, local, iganet::west),
               std::invalid_argument);
}

TEST(MultiPatch, SerializesPatchesInterfacesAndDerivedBoundariesToXml) {
  using patch_type = iganet::BSplinePatch<double, 2, 2>;
  using spline_type = iganet::UniformBSpline<double, 2, 1, 1>;

  iganet::MultiPatch<patch_type> patches;
  patches.addPatch(
      std::make_shared<spline_type>(spline_type(std::array<int64_t, 2>{2, 2})));
  patches.addPatch(
      std::make_shared<spline_type>(spline_type(std::array<int64_t, 2>{2, 2})));
  patches.addInterface(0, iganet::east, 1, iganet::west);

  const auto document = patches.to_xml(7, "domain", 3);
  const auto root = document.child("xml");
  EXPECT_TRUE(root.find_child_by_attribute("Geometry", "id", "0"));
  EXPECT_TRUE(root.find_child_by_attribute("Geometry", "id", "1"));

  const auto multiPatch = root.child("MultiPatch");
  ASSERT_TRUE(multiPatch);
  EXPECT_EQ(multiPatch.attribute("parDim").as_int(), 2);
  EXPECT_EQ(multiPatch.attribute("id").as_int(), 7);
  EXPECT_STREQ(multiPatch.attribute("label").value(), "domain");
  EXPECT_EQ(multiPatch.attribute("index").as_int(), 3);
  EXPECT_STREQ(multiPatch.child("patches").attribute("type").value(),
               "id_range");
  EXPECT_STREQ(multiPatch.child_value("patches"), "0 1");
  EXPECT_STREQ(multiPatch.child_value("interfaces"),
               "\n      0 2 1 1 0 1 1 1\n    ");
  EXPECT_STREQ(multiPatch.child_value("boundary"),
               "\n      0 1\n      0 3\n      0 4\n      1 2\n      1 3\n      "
               "1 4\n    ");

  iganet::MultiPatch<patch_type> restored;
  restored.addPatch(
      std::make_shared<spline_type>(spline_type(std::array<int64_t, 2>{2, 2})));
  restored.addPatch(
      std::make_shared<spline_type>(spline_type(std::array<int64_t, 2>{2, 2})));
  EXPECT_EQ(&restored.from_xml(document, 7, "domain", 3), &restored);
  ASSERT_EQ(restored.ninterfaces(), 1);
  EXPECT_EQ(&restored.interface(0).firstPatch(), &restored.patch(0));
  EXPECT_EQ(restored.interface(0).firstSide(), iganet::east);
  EXPECT_EQ(&restored.interface(0).secondPatch(), &restored.patch(1));
  EXPECT_EQ(restored.interface(0).secondSide(), iganet::west);
}

TEST(MultiPatch, SerializesAndRestoresJson) {
  using patch_type = iganet::BSplinePatch<double, 2, 2>;
  using spline_type = iganet::UniformBSpline<double, 2, 1, 1>;

  iganet::MultiPatch<patch_type> patches;
  patches.addPatch(
      std::make_shared<spline_type>(spline_type(std::array<int64_t, 2>{2, 2})));
  patches.addPatch(
      std::make_shared<spline_type>(spline_type(std::array<int64_t, 2>{2, 2})));
  patches.addInterface(0, iganet::north, 1, iganet::south);

  const auto json = patches.to_json();
  EXPECT_EQ(json["parDim"], 2);
  ASSERT_EQ(json["patches"].size(), 2);
  ASSERT_EQ(json["interfaces"].size(), 1);
  EXPECT_EQ(json["interfaces"][0]["patches"], nlohmann::json({0, 1}));
  EXPECT_EQ(json["interfaces"][0]["sides"],
            nlohmann::json({iganet::north, iganet::south}));
  EXPECT_EQ(json["interfaces"][0]["direction"], nlohmann::json({0, 1}));
  EXPECT_EQ(json["interfaces"][0]["orientation"], nlohmann::json({1, 1}));
  EXPECT_EQ(json["boundaries"].size(), 6);

  iganet::MultiPatch<patch_type> restored;
  restored.addPatch(
      std::make_shared<spline_type>(spline_type(std::array<int64_t, 2>{2, 2})));
  restored.addPatch(
      std::make_shared<spline_type>(spline_type(std::array<int64_t, 2>{2, 2})));
  EXPECT_EQ(&restored.from_json(json), &restored);
  ASSERT_EQ(restored.ninterfaces(), 1);
  EXPECT_EQ(restored.interface(0).firstSide(), iganet::north);
  EXPECT_EQ(restored.interface(0).secondSide(), iganet::south);
}

#ifndef NDEBUG
TEST(MultiPatchDeathTest, RejectsOutOfRangePatchAccess) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  iganet::MultiPatch<PatchValue> patches;
  EXPECT_DEATH_IF_SUPPORTED((void)patches.patch(0), "");
}
#endif
