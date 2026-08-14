/**
   @file unittests/src/unittest_matrix.cxx

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

TEST(Matrix, CombinesTwoAndThreeDimensionalIndices) {
  iganet::utils::TensorArray<2> indices2{
      torch::tensor({{0, 1}, {0, 1}}, torch::kInt64),
      torch::tensor({{0, 1}, {0, 1}}, torch::kInt64)};
  auto matrix2 = iganet::utils::to_sparseCsrTensor<2>(
      indices2, {2, 2}, torch::ones({2, 4}), {2, 4});
  EXPECT_TRUE(torch::equal(matrix2.to_dense(), torch::ones({2, 4})));

  iganet::utils::TensorArray<3> indices3{
      torch::tensor({{0, 1}}, torch::kInt64),
      torch::tensor({{0, 1}}, torch::kInt64),
      torch::tensor({{0, 1}}, torch::kInt64)};
  auto matrix3 = iganet::utils::to_sparseCsrTensor<3>(
      indices3, {2, 2, 2}, torch::ones({1, 8}), {1, 8});
  EXPECT_TRUE(torch::equal(matrix3.to_dense(), torch::ones({1, 8})));
}

TEST(Matrix, BuildsIndicesFromKnotsAndDegrees) {
  iganet::utils::TensorArray<1> knots{
      torch::tensor({1, 2}, torch::kInt64)};
  auto matrix = iganet::utils::to_sparseCsrTensor<1>(
      knots, std::array<short, 1>{1}, std::array<int64_t, 1>{3},
      torch::tensor({{1., 2.}, {3., 4.}}), {2, 3});
  EXPECT_TRUE(torch::equal(matrix.to_dense(),
                           torch::tensor({{1., 2., 0.}, {0., 3., 4.}})));
}

TEST(Matrix, RejectsInvalidDimensionAndSize) {
  EXPECT_ANY_THROW(iganet::utils::to_sparseCsrTensor(
      torch::tensor({{0, 1}}, torch::kInt64), torch::ones({1, 2}), {2}));
  EXPECT_ANY_THROW(iganet::utils::to_sparseCsrTensor(
      torch::tensor({{0}, {1}, {0}}, torch::kInt64), torch::ones({3, 1}),
      {2, 2}));
  iganet::utils::TensorArray<4> invalid{
      torch::zeros({1, 1}, torch::kInt64), torch::zeros({1, 1}, torch::kInt64),
      torch::zeros({1, 1}, torch::kInt64), torch::zeros({1, 1}, torch::kInt64)};
  EXPECT_THROW((iganet::utils::to_sparseCsrTensor<4>(
                   invalid, {1, 1, 1, 1}, torch::ones({1, 1}), {1, 1})),
               std::runtime_error);
}

TEST(MatrixProperties, SparseCsrAgreesWithIndependentDenseConstruction) {
  torch::manual_seed(161803);
  constexpr int64_t matrix_rows = 7;
  constexpr int64_t matrix_cols = 9;
  constexpr int64_t populated_rows = 5;
  constexpr int64_t entries_per_row = 4;

  for (int sample = 0; sample < 20; ++sample) {
    auto columns = torch::empty({populated_rows, entries_per_row},
                                torch::kInt64);
    auto values = torch::randn({populated_rows, entries_per_row},
                               torch::kFloat64);
    auto expected = torch::zeros({matrix_rows, matrix_cols}, torch::kFloat64);

    for (int64_t row = 0; row < populated_rows; ++row) {
      auto row_columns = torch::randperm(matrix_cols, torch::kInt64)
                             .slice(0, 0, entries_per_row);
      columns[row].copy_(row_columns);
      for (int64_t entry = 0; entry < entries_per_row; ++entry)
        expected.index_put_({row, row_columns[entry]}, values[row][entry]);
    }

    auto sparse = iganet::utils::to_sparseCsrTensor(
        columns, values, {matrix_rows, matrix_cols});
    EXPECT_TRUE(torch::equal(sparse.to_dense(), expected));
  }
}
