/**
   @file unittests/unittest_nurbs.cxx

   @brief NURBS unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

#include <sstream>

TEST(Nurbs, UniformConstructionInitializesHomogeneousWeights) {
  using Nurbs = iganet::UniformNurbs<double, 2, 2>;
  Nurbs nurbs({3});
  EXPECT_EQ(Nurbs::parDim(), 1);
  EXPECT_EQ(Nurbs::geoDim(), 2);
  EXPECT_TRUE(Nurbs::is_uniform());
  EXPECT_FALSE(Nurbs::is_nonuniform());
  EXPECT_EQ(nurbs.weights().numel(), 3);
  EXPECT_TRUE(torch::equal(nurbs.weights(),
                           torch::ones(3, torch::kFloat64)));
  EXPECT_TRUE(torch::equal(nurbs.weights(), nurbs.coeffs(2)));

  nurbs.weights().index_put_({1}, 2.0);
  EXPECT_DOUBLE_EQ(nurbs.coeffs(2)[1].item<double>(), 2.0);

  std::ostringstream stream;
  stream << nurbs;
  EXPECT_FALSE(stream.str().empty());
}

TEST(Nurbs, UniformConstructionSupportsExplicitHomogeneousCoefficients) {
  iganet::utils::TensorArray3 coeffs{
      torch::tensor({0., 2.}, torch::kFloat64),
      torch::tensor({1., 3.}, torch::kFloat64),
      torch::tensor({1., 4.}, torch::kFloat64)};
  iganet::UniformNurbs<double, 2, 1> nurbs({2}, coeffs, true);
  EXPECT_TRUE(torch::equal(nurbs.coeffs(0), coeffs[0]));
  EXPECT_TRUE(torch::equal(nurbs.coeffs(1), coeffs[1]));
  EXPECT_TRUE(torch::equal(nurbs.weights(), coeffs[2]));

  coeffs[2].fill_(9.0);
  EXPECT_FALSE(torch::equal(nurbs.weights(), coeffs[2]));
}

TEST(Nurbs, NonUniformTypeTraitsReportKnotLayout) {
  using Nurbs = iganet::NonUniformNurbs<double, 2, 1>;
  static_assert(Nurbs::geoDim() == 2);
  static_assert(Nurbs::parDim() == 1);
  EXPECT_FALSE(Nurbs::is_uniform());
  EXPECT_TRUE(Nurbs::is_nonuniform());
}
