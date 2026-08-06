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

TEST(TensorArray, OptionsAndShapeOptionsOverloads) {
  auto options = iganet::Options<double>{}.requires_grad(true);
  auto tensors = iganet::utils::to_tensorArray(options, {1., 2.}, {3., 4.});
  EXPECT_TRUE(tensors[0].requires_grad());
  EXPECT_EQ(tensors[1].scalar_type(), torch::kFloat64);

  const std::array<int64_t, 2> shape_storage{1, 2};
  auto shaped = iganet::utils::to_tensorArray(
      torch::IntArrayRef{shape_storage}, options, {1., 2.}, {3., 4.});
  EXPECT_EQ(shaped[0].sizes(), (torch::IntArrayRef{1, 2}));
  EXPECT_TRUE(shaped[1].requires_grad());
}

TEST(TensorArray, BlockTensorAndDeviceAwareAccessorOverloads) {
  iganet::utils::BlockTensor<torch::Tensor, 1, 2> blocks(
      torch::tensor({{1., 2.}}, torch::kFloat64),
      torch::tensor({{3., 4.}}, torch::kFloat64));
  auto [owned_blocks, block_accessors] =
      iganet::utils::to_tensorAccessor<double, 2>(blocks, torch::kCPU);
  EXPECT_DOUBLE_EQ(block_accessors[0][0][1], 2.0);
  EXPECT_DOUBLE_EQ(block_accessors[1][0][0], 3.0);

  iganet::utils::TensorArray<2> tensors{
      torch::tensor({{5., 6.}}, torch::kFloat64),
      torch::tensor({{7., 8.}}, torch::kFloat64)};
  auto [owned, accessors] =
      iganet::utils::to_tensorAccessor<double, 2>(tensors, torch::kCPU);
  EXPECT_DOUBLE_EQ(accessors[0][0][0], 5.0);
  EXPECT_DOUBLE_EQ(accessors[1][0][1], 8.0);
}
