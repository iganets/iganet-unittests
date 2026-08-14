/**
   @file unittests/src/unittest_linalg.cxx

   @brief Linear algebra utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

TEST(Linalg, DotAndKroneckerProducts) {
  auto a = torch::tensor({{1., 2.}, {3., 4.}});
  auto b = torch::tensor({{2., 3.}, {4., 5.}});
  EXPECT_TRUE(torch::equal(iganet::utils::dotproduct<0>(a, b),
                           torch::tensor({14., 26.})));
  EXPECT_TRUE(torch::equal(iganet::utils::dotproduct<1>(a, b),
                           torch::tensor({8., 32.})));

  auto x = torch::tensor({1., 2.});
  auto y = torch::tensor({3., 4., 5.});
  EXPECT_TRUE(torch::equal(iganet::utils::kronproduct(x, y),
                           torch::tensor({3., 4., 5., 6., 8., 10.})));
  EXPECT_TRUE(torch::equal(iganet::utils::kron(x, y), torch::kron(x, y)));
  EXPECT_TRUE(torch::equal(iganet::utils::kron(x, y, x),
                           torch::kron(x, torch::kron(y, x))));
}

TEST(Linalg, PartialArrayReductions) {
  std::array<int, 4> values{2, 3, 5, 7};
  EXPECT_EQ(iganet::utils::prod(values), 210);
  EXPECT_EQ(iganet::utils::prod(values, 1, 2), 15);
  EXPECT_EQ(iganet::utils::sum(values), 17);
  EXPECT_EQ(iganet::utils::sum(values, 1, 2), 8);
}

TEST(Linalg, DirectionalAndVariadicKroneckerProducts) {
  auto a = torch::tensor({{1., 2.}, {3., 4.}});
  auto b = torch::tensor({{5., 6.}, {7., 8.}});
  auto along_rows = iganet::utils::kronproduct<0>(a, b);
  auto along_cols = iganet::utils::kronproduct<1>(a, b);
  EXPECT_EQ(along_rows.sizes(), (torch::IntArrayRef{4, 2}));
  EXPECT_EQ(along_cols.sizes(), (torch::IntArrayRef{2, 4}));
  EXPECT_TRUE(torch::equal(iganet::utils::kronproduct(
                               torch::tensor({1., 2.}), torch::tensor({3., 4.}),
                               torch::tensor({5., 6.})),
                           torch::tensor({15., 18., 20., 24., 30., 36., 40., 48.})));
}

TEST(Linalg, RejectsUnsupportedTensorRank) {
  auto a = torch::ones({1, 1, 1, 1, 1, 1, 1, 1, 1});
  EXPECT_THROW(iganet::utils::kronproduct(a, a), std::runtime_error);
}

TEST(LinalgProperties, VariadicKronAgreesWithNestedTorchKron) {
  torch::manual_seed(314159);
  for (int sample = 0; sample < 20; ++sample) {
    auto a = torch::randn({2 + sample % 2}, torch::kFloat64);
    auto b = torch::randn({2, 2}, torch::kFloat64);
    auto c = torch::randn({1 + sample % 3}, torch::kFloat64);
    auto actual = iganet::utils::kron(a, b, c);
    auto expected = torch::kron(a, torch::kron(b, c));
    EXPECT_TRUE(torch::equal(actual, expected));
  }
}
