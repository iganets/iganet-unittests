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
