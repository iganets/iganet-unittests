/**
   @file unittests/unittest_solver.cxx

   @brief Solver utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

#include <limits>

TEST(Solver, ConjugateGradientSolvesSpdSystem) {
  auto A = torch::tensor({{4., 1.}, {1., 3.}}, torch::kFloat64);
  auto b = torch::tensor({1., 2.}, torch::kFloat64);
  auto [x, iterations, residual] = iganet::utils::solve_cg(A, b, 20, 1e-12);
  EXPECT_TRUE(torch::allclose(x, torch::tensor({1. / 11, 7. / 11}, torch::kFloat64), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);
}

TEST(Solver, BiCgStabSolvesNonsymmetricSystem) {
  auto A = torch::tensor({{4., 1.}, {2., 3.}}, torch::kFloat64);
  auto b = torch::tensor({1., 2.}, torch::kFloat64);
  auto [x, iterations, residual] = iganet::utils::solve_bicgstab(A, b, 20, 1e-12);
  EXPECT_TRUE(torch::allclose(x, torch::tensor({0.1, 0.6}, torch::kFloat64), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);
}

TEST(Solver, BiCgStabSupportsSparseCsrMatrices) {
  auto dense = torch::tensor({{4., 1.}, {2., 3.}}, torch::kFloat64);
  auto A = dense.to_sparse_csr();
  auto b = torch::tensor({1., 2.}, torch::kFloat64);

  auto [x, iterations, residual] =
      iganet::utils::solve_bicgstab(A, b, 20, 1e-12);

  EXPECT_TRUE(torch::allclose(
      x, torch::tensor({0.1, 0.6}, torch::kFloat64), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);

  auto invalid_values = A.values().clone();
  invalid_values.index_put_({0}, std::numeric_limits<double>::infinity());
  auto invalid = torch::sparse_csr_tensor(A.crow_indices(), A.col_indices(),
                                          invalid_values, A.sizes(),
                                          invalid_values.options().layout(
                                              torch::kSparseCsr));
  EXPECT_THROW((void)iganet::utils::solve_bicgstab(invalid, b), c10::Error);
}

TEST(Solver, HandlesZeroRightHandSideAndIterationLimit) {
  auto A = torch::eye(2, torch::kFloat64);
  auto zero = torch::zeros(2, torch::kFloat64);
  auto [x0, iteration0, residual0] = iganet::utils::solve_cg(A, zero);
  EXPECT_TRUE(torch::equal(x0, zero));
  EXPECT_EQ(iteration0, -1);
  EXPECT_DOUBLE_EQ(residual0, 0.0);

  auto [x, iterations, residual] =
      iganet::utils::solve_cg(A, torch::ones(2, torch::kFloat64), 0);
  EXPECT_EQ(iterations, 0);
  EXPECT_GT(residual, 0.0);
}

TEST(Solver, BiCgStabHandlesZeroRightHandSideAndIterationLimit) {
  auto A = torch::eye(3, torch::kFloat64);
  auto zero = torch::zeros(3, torch::kFloat64);
  auto [x0, iteration0, residual0] =
      iganet::utils::solve_bicgstab(A, zero);
  EXPECT_TRUE(torch::equal(x0, zero));
  EXPECT_EQ(iteration0, -1);
  EXPECT_DOUBLE_EQ(residual0, 0.0);

  auto [x, iterations, residual] = iganet::utils::solve_bicgstab(
      A, torch::ones(3, torch::kFloat64), 0);
  EXPECT_EQ(iterations, 0);
  EXPECT_GT(residual, 0.0);
}

TEST(Solver, SupportsFloatInputsAndLargerSystems) {
  auto A = torch::diag(torch::tensor({1.f, 2.f, 4.f, 8.f}));
  auto b = torch::tensor({1.f, 2.f, 4.f, 8.f});
  auto [x, iterations, residual] = iganet::utils::solve_cg(A, b, 20, 1e-5);
  EXPECT_TRUE(torch::allclose(x, torch::ones(4), 1e-4, 1e-4));
  EXPECT_LT(residual, 1e-5);
}

TEST(Solver, RejectsSingularSystemsOnNumericalBreakdown) {
  auto singular = torch::zeros({2, 2}, torch::kFloat64);
  auto b = torch::ones(2, torch::kFloat64);
  EXPECT_THROW((void)iganet::utils::solve_cg(singular, b), c10::Error);
  EXPECT_THROW((void)iganet::utils::solve_bicgstab(singular, b), c10::Error);
}

TEST(Solver, RejectsNanAndInfiniteInputs) {
  auto identity = torch::eye(2, torch::kFloat64);
  auto finite_b = torch::ones(2, torch::kFloat64);

  auto nan_matrix = identity.clone();
  nan_matrix.index_put_({0, 0}, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW((void)iganet::utils::solve_cg(nan_matrix, finite_b), c10::Error);

  auto inf_b = finite_b.clone();
  inf_b.index_put_({1}, std::numeric_limits<double>::infinity());
  EXPECT_THROW((void)iganet::utils::solve_bicgstab(identity, inf_b),
               c10::Error);
}

TEST(Solver, RejectsInvalidShapesAndParameters) {
  auto identity = torch::eye(2, torch::kFloat64);
  auto b = torch::ones(2, torch::kFloat64);
  EXPECT_THROW((void)iganet::utils::solve_cg(torch::ones({2, 3}), b),
               c10::Error);
  EXPECT_THROW((void)iganet::utils::solve_bicgstab(identity, torch::ones(3)),
               c10::Error);
  EXPECT_THROW((void)iganet::utils::solve_cg(identity, b, -1), c10::Error);
  EXPECT_THROW((void)iganet::utils::solve_bicgstab(identity, b, 10, 0.0),
               c10::Error);
}
