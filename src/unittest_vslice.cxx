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
  EXPECT_TRUE(torch::equal(iganet::utils::VSlice(index, start, stop, leading),
                           torch::tensor({3, 4, 6, 7})));
  EXPECT_TRUE(torch::equal(iganet::utils::VSlice<true>(index, start, stop, leading),
                           torch::tensor({3, 4, 6, 7})));
}
