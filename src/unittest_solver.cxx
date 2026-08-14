/**
   @file unittests/src/unittest_solver.cxx

   @brief Solver utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet/iganet.h>
#include <gtest/gtest.h>

#include <limits>

namespace {

struct ValidPreconditioner {
  torch::Tensor operator()(const torch::Tensor &residual) {
    return residual.clone();
  }
};

struct WrongArgumentPreconditioner {
  torch::Tensor operator()(int) { return {}; }
};

struct WrongResultPreconditioner {
  double operator()(const torch::Tensor &) { return 0.0; }
};

static_assert(
    iganet::utils::IterativeSolverPreconditioner<ValidPreconditioner>);
static_assert(
    !iganet::utils::IterativeSolverPreconditioner<WrongArgumentPreconditioner>);
static_assert(
    !iganet::utils::IterativeSolverPreconditioner<WrongResultPreconditioner>);

} // namespace

TEST(Solver, ConjugateGradientSolvesSpdSystem) {
  auto A = torch::tensor({{4., 1.}, {1., 3.}}, torch::kFloat64);
  auto b = torch::tensor({1., 2.}, torch::kFloat64);
  auto [x, iterations, residual] = iganet::utils::cg(A, b, 20, 1e-12);
  EXPECT_TRUE(torch::allclose(x, torch::tensor({1. / 11, 7. / 11}, torch::kFloat64), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);
}

TEST(Solver, BiCgStabSolvesNonsymmetricSystem) {
  auto A = torch::tensor({{4., 1.}, {2., 3.}}, torch::kFloat64);
  auto b = torch::tensor({1., 2.}, torch::kFloat64);
  auto [x, iterations, residual] = iganet::utils::bicgstab(A, b, 20, 1e-12);
  EXPECT_TRUE(torch::allclose(x, torch::tensor({0.1, 0.6}, torch::kFloat64), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);
}

TEST(Solver, BiCgStabSupportsSparseCsrMatrices) {
  auto dense = torch::tensor({{4., 1.}, {2., 3.}}, torch::kFloat64);
  auto A = dense.to_sparse_csr();
  auto b = torch::tensor({1., 2.}, torch::kFloat64);

  auto [x, iterations, residual] =
      iganet::utils::bicgstab(A, b, 20, 1e-12);

  EXPECT_TRUE(torch::allclose(
      x, torch::tensor({0.1, 0.6}, torch::kFloat64), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);

#ifndef NDEBUG
  auto invalid_values = A.values().clone();
  invalid_values.index_put_({0}, std::numeric_limits<double>::infinity());
  auto invalid = torch::sparse_csr_tensor(A.crow_indices(), A.col_indices(),
                                          invalid_values, A.sizes(),
                                          invalid_values.options().layout(
                                              torch::kSparseCsr));
  EXPECT_THROW((void)iganet::utils::bicgstab(invalid, b), c10::Error);
#endif
}

TEST(Solver, PreconditionedConjugateGradientUsesJacobiPreconditioner) {
  auto diagonal = torch::tensor({1., 10., 100., 1000.}, torch::kFloat64);
  auto A = torch::diag(diagonal);
  auto b = torch::ones(4, torch::kFloat64);
  auto jacobi = [&diagonal](const torch::Tensor &r) {
    return r / diagonal;
  };

  auto [x, iterations, residual] =
      iganet::utils::pcg(A, b, jacobi, 20, 1e-12);

  EXPECT_TRUE(torch::allclose(x, diagonal.reciprocal(), 1e-12, 1e-12));
  EXPECT_EQ(iterations, 0);
  EXPECT_LT(residual, 1e-12);
}

TEST(Solver, PreconditionedBiCgStabSupportsSparseCsrMatrices) {
  auto dense = torch::tensor({{4., 1., 0.},
                              {2., 3., 1.},
                              {0., 1., 2.}},
                             torch::kFloat64);
  auto A = dense.to_sparse_csr();
  auto b = torch::tensor({1., 2., 3.}, torch::kFloat64);
  auto diagonal = dense.diag();
  auto jacobi = [&diagonal](const torch::Tensor &r) {
    return r / diagonal;
  };

  auto [x, iterations, residual] =
      iganet::utils::pbicgstab(A, b, jacobi, 30, 1e-12);

  EXPECT_TRUE(torch::allclose(x, torch::linalg_solve(dense, b), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);
}

TEST(Solver, IdentityPreconditionerAgreesWithUnpreconditionedSolvers) {
  auto spd = torch::tensor({{4., 1.}, {1., 3.}}, torch::kFloat64);
  auto nonsymmetric = torch::tensor({{4., 1.}, {2., 3.}}, torch::kFloat64);
  auto b = torch::tensor({1., 2.}, torch::kFloat64);
  auto identity = [](const torch::Tensor &r) { return r.clone(); };

  auto [cg_x, cg_iterations, cg_residual] =
      iganet::utils::cg(spd, b, 20, 1e-12);
  auto [pcg_x, pcg_iterations, pcg_residual] =
      iganet::utils::pcg(spd, b, identity, 20, 1e-12);
  EXPECT_TRUE(torch::allclose(pcg_x, cg_x, 1e-12, 1e-12));
  EXPECT_EQ(pcg_iterations, cg_iterations);
  EXPECT_NEAR(pcg_residual, cg_residual, 1e-14);

  auto [bicg_x, bicg_iterations, bicg_residual] =
      iganet::utils::bicgstab(nonsymmetric, b, 20, 1e-12);
  auto [pbicg_x, pbicg_iterations, pbicg_residual] =
      iganet::utils::pbicgstab(nonsymmetric, b, identity, 20, 1e-12);
  EXPECT_TRUE(torch::allclose(pbicg_x, bicg_x, 1e-12, 1e-12));
  EXPECT_EQ(pbicg_iterations, bicg_iterations);
  EXPECT_NEAR(pbicg_residual, bicg_residual, 1e-14);
}

TEST(Solver, PreconditionedSolversHandleZeroRightHandSideAndIterationLimit) {
  auto A = torch::eye(2, torch::kFloat64);
  auto zero = torch::zeros(2, torch::kFloat64);
  int calls = 0;
  auto preconditioner = [&calls](const torch::Tensor &r) {
    ++calls;
    return r.clone();
  };

  auto [pcg_x, pcg_iteration, pcg_residual] =
      iganet::utils::pcg(A, zero, preconditioner);
  EXPECT_TRUE(torch::equal(pcg_x, zero));
  EXPECT_EQ(pcg_iteration, -1);
  EXPECT_DOUBLE_EQ(pcg_residual, 0.0);

  auto [pbicg_x, pbicg_iteration, pbicg_residual] =
      iganet::utils::pbicgstab(A, torch::ones(2, torch::kFloat64),
                                    preconditioner, 0);
  EXPECT_TRUE(torch::equal(pbicg_x, zero));
  EXPECT_EQ(pbicg_iteration, 0);
  EXPECT_GT(pbicg_residual, 0.0);
  EXPECT_EQ(calls, 0);
}

TEST(Solver, TensorPreconditionerOverloadsAgreeWithCallableOverloads) {
  auto spd = torch::tensor({{4., 1.}, {1., 3.}}, torch::kFloat64);
  auto nonsymmetric = torch::tensor({{4., 1.}, {2., 3.}}, torch::kFloat64);
  auto b = torch::tensor({1., 2.}, torch::kFloat64);
  auto inverse = torch::diag(spd.diag().reciprocal());
  auto apply = [&inverse](const torch::Tensor &r) {
    return inverse.matmul(r);
  };

  auto [pcg_tensor, pcg_tensor_iterations, pcg_tensor_residual] =
      iganet::utils::pcg(spd, b, inverse, 20, 1e-12);
  auto [pcg_callable, pcg_callable_iterations, pcg_callable_residual] =
      iganet::utils::pcg(spd, b, apply, 20, 1e-12);
  EXPECT_TRUE(torch::allclose(pcg_tensor, pcg_callable, 1e-12, 1e-12));
  EXPECT_EQ(pcg_tensor_iterations, pcg_callable_iterations);
  EXPECT_NEAR(pcg_tensor_residual, pcg_callable_residual, 1e-14);

  auto [pbicg_tensor, pbicg_tensor_iterations, pbicg_tensor_residual] =
      iganet::utils::pbicgstab(nonsymmetric, b, inverse, 20, 1e-12);
  auto [pbicg_callable, pbicg_callable_iterations,
        pbicg_callable_residual] =
      iganet::utils::pbicgstab(nonsymmetric, b, apply, 20, 1e-12);
  EXPECT_TRUE(torch::allclose(pbicg_tensor, pbicg_callable, 1e-12, 1e-12));
  EXPECT_EQ(pbicg_tensor_iterations, pbicg_callable_iterations);
  EXPECT_NEAR(pbicg_tensor_residual, pbicg_callable_residual, 1e-14);
}

TEST(Solver, MinresSolvesSymmetricIndefiniteSystem) {
  auto A = torch::tensor({{0., 2., 0.},
                          {2., 0., 1.},
                          {0., 1., -1.}},
                         torch::kFloat64);
  auto b = torch::tensor({1., 2., 3.}, torch::kFloat64);

  auto [x, iterations, residual] =
      iganet::utils::minres(A, b, 20, 1e-12);

  EXPECT_TRUE(torch::allclose(x, torch::linalg_solve(A, b), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);
}

TEST(Solver, MinresSupportsSparseCsrMatrices) {
  auto dense = torch::tensor({{2., 1., 0.},
                              {1., -1., 1.},
                              {0., 1., 3.}},
                             torch::kFloat64);
  auto A = dense.to_sparse_csr();
  auto b = torch::tensor({1., 2., 3.}, torch::kFloat64);

  auto [x, iterations, residual] =
      iganet::utils::minres(A, b, 20, 1e-12);

  EXPECT_TRUE(torch::allclose(x, torch::linalg_solve(dense, b), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);
}

TEST(Solver, PreconditionedMinresSupportsCallableAndTensorPreconditioners) {
  auto diagonal = torch::tensor({-1., 2., 100.}, torch::kFloat64);
  auto A = torch::diag(diagonal);
  auto b = torch::ones(3, torch::kFloat64);
  auto inverse_diagonal = diagonal.abs().reciprocal();
  auto inverse = torch::diag(inverse_diagonal);
  auto jacobi = [&inverse_diagonal](const torch::Tensor &r) {
    return r * inverse_diagonal;
  };

  auto [callable_x, callable_iterations, callable_residual] =
      iganet::utils::pminres(A, b, jacobi, 20, 1e-12);
  auto [tensor_x, tensor_iterations, tensor_residual] =
      iganet::utils::pminres(A, b, inverse, 20, 1e-12);

  EXPECT_TRUE(torch::allclose(callable_x, diagonal.reciprocal(), 1e-10,
                              1e-10));
  EXPECT_TRUE(torch::allclose(tensor_x, callable_x, 1e-12, 1e-12));
  EXPECT_EQ(tensor_iterations, callable_iterations);
  EXPECT_LT(callable_residual, 1e-12);
  EXPECT_LT(tensor_residual, 1e-12);
}

TEST(Solver, MinresHandlesZeroRightHandSideAndIterationLimit) {
  auto A = torch::diag(torch::tensor({-1., 2.}, torch::kFloat64));
  auto zero = torch::zeros(2, torch::kFloat64);
  auto [zero_x, zero_iteration, zero_residual] =
      iganet::utils::minres(A, zero);
  EXPECT_TRUE(torch::equal(zero_x, zero));
  EXPECT_EQ(zero_iteration, -1);
  EXPECT_DOUBLE_EQ(zero_residual, 0.0);

  auto [x, iteration, residual] =
      iganet::utils::minres(A, torch::ones(2, torch::kFloat64), 0);
  EXPECT_TRUE(torch::equal(x, zero));
  EXPECT_EQ(iteration, 0);
  EXPECT_GT(residual, 0.0);
}

TEST(Solver, GmresSolvesNonsymmetricSystemAcrossRestarts) {
  auto A = torch::tensor({{4., 1., 0., 0.},
                          {2., 3., 1., 0.},
                          {0., 1., 2., 1.},
                          {1., 0., 1., 3.}},
                         torch::kFloat64);
  auto b = torch::tensor({1., 2., 3., 4.}, torch::kFloat64);

  auto [x, iterations, residual] =
      iganet::utils::gmres(A, b, 40, 1e-11, 2);

  EXPECT_TRUE(torch::allclose(x, torch::linalg_solve(A, b), 1e-9, 1e-9));
  EXPECT_GE(iterations, 2);
  EXPECT_LT(residual, 1e-11);
}

TEST(Solver, GmresSupportsSparseCsrMatrices) {
  auto dense = torch::tensor({{4., 1., 0.},
                              {2., 3., 1.},
                              {0., 1., 2.}},
                             torch::kFloat64);
  auto A = dense.to_sparse_csr();
  auto b = torch::tensor({1., 2., 3.}, torch::kFloat64);

  auto [x, iterations, residual] =
      iganet::utils::gmres(A, b, 20, 1e-12, 3);

  EXPECT_TRUE(torch::allclose(x, torch::linalg_solve(dense, b), 1e-10, 1e-10));
  EXPECT_GE(iterations, 0);
  EXPECT_LT(residual, 1e-12);
}

TEST(Solver, FlexibleGmresSupportsCallableAndTensorPreconditioners) {
  auto diagonal = torch::tensor({1., 10., 100., 1000.}, torch::kFloat64);
  auto A = torch::diag(diagonal);
  auto b = torch::ones(4, torch::kFloat64);
  auto inverse = torch::diag(diagonal.reciprocal());
  auto jacobi = [&diagonal](const torch::Tensor &r) {
    return r / diagonal;
  };

  auto [callable_x, callable_iterations, callable_residual] =
      iganet::utils::fgmres(A, b, jacobi, 10, 1e-12, 4);
  auto [tensor_x, tensor_iterations, tensor_residual] =
      iganet::utils::fgmres(A, b, inverse, 10, 1e-12, 4);

  EXPECT_TRUE(torch::allclose(callable_x, diagonal.reciprocal(), 1e-12,
                              1e-12));
  EXPECT_TRUE(torch::allclose(tensor_x, callable_x, 1e-12, 1e-12));
  EXPECT_EQ(callable_iterations, 0);
  EXPECT_EQ(tensor_iterations, callable_iterations);
  EXPECT_LT(callable_residual, 1e-12);
  EXPECT_LT(tensor_residual, 1e-12);
}

TEST(Solver, GmresHandlesZeroRightHandSideAndIterationLimit) {
  auto A = torch::eye(2, torch::kFloat64);
  auto zero = torch::zeros(2, torch::kFloat64);
  auto [zero_x, zero_iteration, zero_residual] =
      iganet::utils::gmres(A, zero);
  EXPECT_TRUE(torch::equal(zero_x, zero));
  EXPECT_EQ(zero_iteration, -1);
  EXPECT_DOUBLE_EQ(zero_residual, 0.0);

  auto [x, iteration, residual] =
      iganet::utils::gmres(A, torch::ones(2, torch::kFloat64), 0);
  EXPECT_TRUE(torch::equal(x, zero));
  EXPECT_EQ(iteration, 0);
  EXPECT_GT(residual, 0.0);
}

TEST(Solver, HandlesZeroRightHandSideAndIterationLimit) {
  auto A = torch::eye(2, torch::kFloat64);
  auto zero = torch::zeros(2, torch::kFloat64);
  auto [x0, iteration0, residual0] = iganet::utils::cg(A, zero);
  EXPECT_TRUE(torch::equal(x0, zero));
  EXPECT_EQ(iteration0, -1);
  EXPECT_DOUBLE_EQ(residual0, 0.0);

  auto [x, iterations, residual] =
      iganet::utils::cg(A, torch::ones(2, torch::kFloat64), 0);
  EXPECT_EQ(iterations, 0);
  EXPECT_GT(residual, 0.0);
}

TEST(Solver, BiCgStabHandlesZeroRightHandSideAndIterationLimit) {
  auto A = torch::eye(3, torch::kFloat64);
  auto zero = torch::zeros(3, torch::kFloat64);
  auto [x0, iteration0, residual0] =
      iganet::utils::bicgstab(A, zero);
  EXPECT_TRUE(torch::equal(x0, zero));
  EXPECT_EQ(iteration0, -1);
  EXPECT_DOUBLE_EQ(residual0, 0.0);

  auto [x, iterations, residual] = iganet::utils::bicgstab(
      A, torch::ones(3, torch::kFloat64), 0);
  EXPECT_EQ(iterations, 0);
  EXPECT_GT(residual, 0.0);
}

TEST(Solver, SupportsFloatInputsAndLargerSystems) {
  auto A = torch::diag(torch::tensor({1.f, 2.f, 4.f, 8.f}));
  auto b = torch::tensor({1.f, 2.f, 4.f, 8.f});
  auto [x, iterations, residual] = iganet::utils::cg(A, b, 20, 1e-5);
  EXPECT_TRUE(torch::allclose(x, torch::ones(4), 1e-4, 1e-4));
  EXPECT_LT(residual, 1e-5);
}

#ifndef NDEBUG
TEST(Solver, RejectsSingularSystemsOnNumericalBreakdown) {
  auto singular = torch::zeros({2, 2}, torch::kFloat64);
  auto b = torch::ones(2, torch::kFloat64);
  EXPECT_THROW((void)iganet::utils::cg(singular, b), c10::Error);
  EXPECT_THROW((void)iganet::utils::bicgstab(singular, b), c10::Error);
}

TEST(Solver, RejectsNanAndInfiniteInputs) {
  auto identity = torch::eye(2, torch::kFloat64);
  auto finite_b = torch::ones(2, torch::kFloat64);

  auto nan_matrix = identity.clone();
  nan_matrix.index_put_({0, 0}, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW((void)iganet::utils::cg(nan_matrix, finite_b), c10::Error);

  auto inf_b = finite_b.clone();
  inf_b.index_put_({1}, std::numeric_limits<double>::infinity());
  EXPECT_THROW((void)iganet::utils::bicgstab(identity, inf_b),
               c10::Error);
}

TEST(Solver, RejectsInvalidShapesAndParameters) {
  auto identity = torch::eye(2, torch::kFloat64);
  auto b = torch::ones(2, torch::kFloat64);
  EXPECT_THROW((void)iganet::utils::cg(torch::ones({2, 3}), b),
               c10::Error);
  EXPECT_THROW((void)iganet::utils::bicgstab(identity, torch::ones(3)),
               c10::Error);
  EXPECT_THROW((void)iganet::utils::cg(identity, b, -1), c10::Error);
  EXPECT_THROW((void)iganet::utils::bicgstab(identity, b, 10, 0.0),
               c10::Error);
}

TEST(Solver, PreconditionedSolversRejectInvalidPreconditionerOutput) {
  auto A = torch::eye(2, torch::kFloat64);
  auto b = torch::ones(2, torch::kFloat64);
  auto wrong_shape = [](const torch::Tensor &r) {
    return torch::ones({r.size(0), 1}, r.options());
  };
  auto nonfinite = [](const torch::Tensor &r) {
    return torch::full_like(r, std::numeric_limits<double>::infinity());
  };

  EXPECT_THROW((void)iganet::utils::pcg(A, b, wrong_shape), c10::Error);
  EXPECT_THROW((void)iganet::utils::pbicgstab(A, b, nonfinite),
               c10::Error);
}

TEST(Solver, TensorPreconditionersAndGmresRejectInvalidParameters) {
  auto A = torch::eye(2, torch::kFloat64);
  auto b = torch::ones(2, torch::kFloat64);
  auto wrong_shape = torch::eye(3, torch::kFloat64);

  EXPECT_THROW((void)iganet::utils::pcg(A, b, wrong_shape), c10::Error);
  EXPECT_THROW((void)iganet::utils::pbicgstab(A, b, wrong_shape),
               c10::Error);
  EXPECT_THROW((void)iganet::utils::gmres(A, b, 10, 1e-10, 0),
               c10::Error);

  auto indefinite_inverse = -torch::eye(2, torch::kFloat64);
  EXPECT_THROW((void)iganet::utils::pminres(A, b, indefinite_inverse),
               c10::Error);
}
#endif
