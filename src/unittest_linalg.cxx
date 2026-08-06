/**
   @file unittests/unittest_linalg.cxx

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
