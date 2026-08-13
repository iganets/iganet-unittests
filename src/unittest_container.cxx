/**
   @file unittests/unittest_container.cxx

   @brief Container utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

using namespace iganet::utils;

TEST(Container, ConstructionAndConversion) {
  EXPECT_EQ((to_array<3>(std::vector<int>{1, 2, 3})),
            (std::array<int, 3>{1, 2, 3}));
  EXPECT_THROW((to_array<2>(std::vector<int>{1})), std::invalid_argument);
  EXPECT_EQ(to_vector(std::array<int, 3>{1, 2, 3}),
            (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(to_array(1, 2.5), (std::array<double, 2>{1.0, 2.5}));
  EXPECT_EQ(to_vector(1, 2, 3), (std::vector<int>{1, 2, 3}));
}

TEST(Container, TensorConversionOwnsDataAndReshapes) {
  std::vector<double> values{1, 2, 3, 4};
  auto tensor = to_tensor(values, {2, 2});
  values[0] = 99;
  EXPECT_TRUE(torch::equal(
    tensor,
    torch::tensor({{1., 2.}, {3., 4.}}, tensor.options())));
  EXPECT_EQ(tensor.scalar_type(), torch::kFloat64);

  auto grad = to_tensor(std::array<float, 2>{1, 2},
                        iganet::Options<float>{}.requires_grad(true));
  EXPECT_TRUE(grad.requires_grad());
}

TEST(Container, ConcatenationAndArrayOperations) {
  EXPECT_EQ(concat(std::array{1, 2}, std::array{3, 4, 5}),
            (std::array{1, 2, 3, 4, 5}));
  EXPECT_EQ(concat(std::vector<int>{1, 2}, std::vector<long>{3, 4}),
            (std::vector<long>{1, 2, 3, 4}));
  EXPECT_EQ((make_array<int, 3>(7)), (std::array{7, 7, 7}));
  EXPECT_EQ((make_array<int>(std::array<double, 2>{1.9, 2.1})),
            (std::array{1, 2}));
  EXPECT_EQ(-(std::array{1, -2}), (std::array{-1, 2}));
  EXPECT_EQ((std::array{1, 2} + std::array{3, 4}), (std::array{4, 6}));
  EXPECT_EQ((std::array{5, 7} - std::array{2, 3}), (std::array{3, 4}));
  EXPECT_EQ((std::array{2, 3} * std::array{4, 5}), (std::array{8, 15}));
  EXPECT_EQ((std::array{8, 15} / std::array{4, 5}), (std::array{2, 3}));
  EXPECT_EQ((remove_from_front<int, 4, 2>({1, 2, 3, 4})),
            (std::array{3, 4}));
  EXPECT_EQ((remove_from_back<int, 4, 2>({1, 2, 3, 4})),
            (std::array{1, 2}));
}

TEST(Container, AppendAndPrepend) {
  EXPECT_EQ((std::array{1, 2} + 3), (std::array{1, 2, 3}));
  EXPECT_EQ((0 + std::array{1, 2}), (std::array{0, 1, 2}));
  EXPECT_EQ((std::vector<int>{1, 2} + 3), (std::vector{1, 2, 3}));
  EXPECT_EQ((0 + std::vector<int>{1, 2}), (std::vector{0, 1, 2}));
}

TEST(Container, ArrayRefAndInitializerListOverloads) {
  std::array<int64_t, 2> sizes{2, 3};
  auto ref = to_ArrayRef(sizes);
  EXPECT_EQ(ref.size(), 2);
  EXPECT_EQ(ref[1], 3);
  EXPECT_EQ(ref + int64_t{4}, (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(int64_t{1} + ref, (std::vector<int64_t>{1, 2, 3}));

  std::initializer_list<double> values{1., 2., 3.};
  auto tensor = to_tensor(values, iganet::Options<double>{});
  EXPECT_TRUE(torch::equal(
    tensor,
    torch::tensor({1., 2., 3.}, tensor.options())));
}

TEST(Container, ConstAndMoveConcatenationOverloads) {
  const std::array<int, 2> a{1, 2};
  const std::array<int, 1> b{3};
  EXPECT_EQ(concat(a, b), (std::array{1, 2, 3}));
  const std::vector<int> x{1, 2};
  const std::vector<double> y{3.5};
  EXPECT_EQ(concat(x, y), (std::vector<double>{1., 2., 3.5}));
}
