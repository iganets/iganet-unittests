/**
   @file unittests/unittest_vslice.cxx

   @brief Vectorized slice utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

TEST(VSlice, OneDimensionalOrders) {
  auto index = torch::tensor({10, 20}, torch::kInt64);
  EXPECT_TRUE(torch::equal(iganet::utils::VSlice(index, 1, 4),
                           torch::tensor({11, 21, 12, 22, 13, 23})));
  EXPECT_TRUE(torch::equal(iganet::utils::VSlice<true>(index, 1, 4),
                           torch::tensor({11, 12, 13, 21, 22, 23})));
}

TEST(VSlice, MultiDimensionalWithLeadingDimension) {
  auto index = iganet::utils::to_tensorArray({0LL}, {1LL});
  std::array<int64_t, 2> start{0, 0};
  std::array<int64_t, 2> stop{2, 2};
  std::array<int64_t, 1> leading{3};
  EXPECT_TRUE(torch::equal(
    iganet::utils::VSlice(index, start, stop, leading),
    torch::tensor({3, 4, 6, 7}, index[0].options())));
  EXPECT_TRUE(torch::equal(
    iganet::utils::VSlice<true>(index, start, stop, leading),
    torch::tensor({3, 4, 6, 7}, index[0].options())));
}

TEST(VSlice, ThreeDimensionsAndMultipleIndices) {
  auto index = iganet::utils::to_tensorArray({0LL, 10LL}, {1LL, 11LL},
                                              {2LL, 12LL});
  std::array<int64_t, 3> start{0, 0, 0};
  std::array<int64_t, 3> stop{2, 2, 2};
  std::array<int64_t, 2> leading{2, 4};
  auto result = iganet::utils::VSlice(index, start, stop, leading);
  std::vector<int64_t> expected;
  for (int64_t k = 0; k < 2; ++k)
    for (int64_t j = 0; j < 2; ++j)
      for (int64_t i = 0; i < 2; ++i) {
        expected.push_back(18 + i + 2 * j + 8 * k);
        expected.push_back(128 + i + 2 * j + 8 * k);
      }
  EXPECT_TRUE(torch::equal(
    std::get<0>(torch::sort(result)),
    std::get<0>(torch::sort(torch::tensor(expected, result.options())))));

  auto transposed = iganet::utils::VSlice<true>(index, start, stop, leading);
  EXPECT_EQ(transposed.numel(), result.numel());
  EXPECT_TRUE(torch::equal(std::get<0>(torch::sort(transposed)),
                           std::get<0>(torch::sort(result))));
}

TEST(VSlice, UsesDefaultLeadingDimension) {
  auto index = iganet::utils::to_tensorArray({1LL}, {2LL});
  EXPECT_TRUE(torch::equal(
    iganet::utils::VSlice(index, std::array<int64_t, 2>{0, 0},
                          std::array<int64_t, 2>{1, 1}),
    torch::tensor({3}, index[0].options())));
}

TEST(VSlice, RejectsInvalidOneDimensionalRanges) {
  auto index = torch::tensor({0, 1}, torch::kInt64);
  EXPECT_THROW((void)iganet::utils::VSlice(index, 2, 2),
               std::invalid_argument);
  EXPECT_THROW((void)iganet::utils::VSlice(index, 3, 1),
               std::invalid_argument);
  EXPECT_THROW((void)iganet::utils::VSlice<true>(index, 0, -1),
               std::invalid_argument);
}

TEST(VSlice, RejectsInvalidMultidimensionalArguments) {
  auto unequal = iganet::utils::to_tensorArray({0LL, 1LL}, {2LL});
  std::array<int64_t, 2> start{0, 0};
  std::array<int64_t, 2> stop{1, 1};
  EXPECT_THROW((void)iganet::utils::VSlice(unequal, start, stop),
               std::invalid_argument);

  auto index = iganet::utils::to_tensorArray({0LL}, {1LL});
  EXPECT_THROW((void)iganet::utils::VSlice(
                   index, std::array<int64_t, 2>{0, 2},
                   std::array<int64_t, 2>{1, 2}),
               std::invalid_argument);
  EXPECT_THROW((void)iganet::utils::VSlice<true>(
                   index, std::array<int64_t, 2>{2, 0},
                   std::array<int64_t, 2>{1, 1}),
               std::invalid_argument);
}
