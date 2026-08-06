/**
   @file unittests/unittest_tensorarray.cxx

   @brief Tensor array utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>
#include <sstream>

TEST(TensorArray, ConstructsWithShapesOptionsAndAccessors) {
  const std::array<int64_t, 2> shape_storage{2, 2};
  const torch::IntArrayRef shape{shape_storage};
  auto tensors = iganet::utils::to_tensorArray(shape, {1., 2., 3., 4.},
                                                {5., 6., 7., 8.});
  ASSERT_EQ(tensors.size(), 2);
  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{2, 2}));
  auto accessors = iganet::utils::to_tensorAccessor<double, 2>(tensors);
  EXPECT_DOUBLE_EQ(accessors[0][1][0], 3.0);
  EXPECT_DOUBLE_EQ(accessors[1][0][1], 6.0);

  auto [owned, accessor] =
      iganet::utils::to_tensorAccessor<double, 2>(tensors[0], torch::kCPU);
  EXPECT_DOUBLE_EQ(accessor[1][1], 4.0);
  EXPECT_TRUE(torch::equal(owned, tensors[0]));
}

TEST(TensorArray, AppliesOperationToEveryTensorAndPrints) {
  auto tensors = iganet::utils::to_tensorArray({-1., 2.}, {-3., 4.});
  auto absolute = TENSORARRAY_FORALL(tensors, abs);
  EXPECT_TRUE(torch::equal(absolute[0], torch::tensor({1., 2.})));
  EXPECT_TRUE(torch::equal(absolute[1], torch::tensor({3., 4.})));
  std::ostringstream stream;
  stream << tensors;
  EXPECT_FALSE(stream.str().empty());
}
