/**
   @file unittests/unittest_matrix.cxx

   @brief Matrix utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

TEST(Matrix, BuildsSparseCsrIncludingEmptyTrailingRows) {
  auto columns = torch::tensor({{0, 2}, {1, 3}}, torch::kInt64);
  auto values = torch::tensor({{1., 2.}, {3., 4.}});
  auto matrix = iganet::utils::to_sparseCsrTensor(columns, values, {3, 4});
  EXPECT_TRUE(torch::equal(matrix.to_dense(),
      torch::tensor({{1., 0., 2., 0.}, {0., 3., 0., 4.}, {0., 0., 0., 0.}})));
  EXPECT_EQ(matrix.layout(), torch::kSparseCsr);
}

TEST(Matrix, ValidatesInputs) {
  EXPECT_ANY_THROW(iganet::utils::to_sparseCsrTensor(
      torch::tensor({0, 1}), torch::ones({2}), {2, 2}));
  EXPECT_ANY_THROW(iganet::utils::to_sparseCsrTensor(
      torch::tensor({{0, 1}}), torch::ones({1, 3}), {2, 2}));
  EXPECT_ANY_THROW(iganet::utils::to_sparseCsrTensor(
      torch::tensor({{0., 1.}}), torch::ones({1, 2}), {2, 2}));
}

TEST(Matrix, CombinesTensorArrayIndices) {
  iganet::utils::TensorArray<1> indices{
      torch::tensor({{0, 1}, {1, 0}}, torch::kInt64)};
  auto values = torch::tensor({{2., 3.}, {4., 5.}});
  auto matrix = iganet::utils::to_sparseCsrTensor<1>(indices, {2}, values,
                                                      {2, 2});
  EXPECT_TRUE(torch::equal(matrix.to_dense(),
                           torch::tensor({{2., 3.}, {5., 4.}})));
}
