/**
   @file unittests/src/unittest_bspline_conversion.cxx

   @brief B-spline conversion unittests

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <common.hpp>
#include <iganet.h>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <unittest_config.hpp>

namespace {

using real_t = iganet::unittests::real_t;
using Uniform = iganet::UniformBSpline<real_t, 1, 2>;
using NonUniform = iganet::NonUniformBSpline<real_t, 1, 2>;

TEST(BSplineConversionTest, UniformToNonUniformConsumesSpline) {
  Uniform spline({5}, iganet::init::linear);
  const auto *knots_data = spline.knots(0).data_ptr<real_t>();
  const auto *coeffs_data = spline.coeffs(0).data_ptr<real_t>();

  auto converted = std::move(spline).to_nonuniform();

  static_assert(std::is_same_v<decltype(converted), NonUniform>);
  EXPECT_TRUE(converted.is_nonuniform());
  EXPECT_EQ(converted.ncoeffs(0), 5);
  EXPECT_EQ(converted.knots(0).data_ptr<real_t>(), knots_data);
  EXPECT_EQ(converted.coeffs(0).data_ptr<real_t>(), coeffs_data);
}

TEST(BSplineConversionTest, NonUniformToUniformConsumesUniformSpline) {
  NonUniform spline({5}, iganet::init::linear);
  const auto *knots_data = spline.knots(0).data_ptr<real_t>();
  const auto *coeffs_data = spline.coeffs(0).data_ptr<real_t>();

  auto converted = std::move(spline).to_uniform();

  static_assert(std::is_same_v<decltype(converted), Uniform>);
  EXPECT_TRUE(converted.is_uniform());
  EXPECT_EQ(converted.ncoeffs(0), 5);
  EXPECT_EQ(converted.knots(0).data_ptr<real_t>(), knots_data);
  EXPECT_EQ(converted.coeffs(0).data_ptr<real_t>(), coeffs_data);
}

TEST(BSplineConversionTest, NonUniformToUniformRejectsNonUniformKnots) {
  NonUniform spline(
      {{{real_t{0}, real_t{0}, real_t{0}, real_t{0.2}, real_t{0.7},
         real_t{1}, real_t{1}, real_t{1}}}},
      iganet::init::linear);

  EXPECT_THROW((void)std::move(spline).to_uniform(), std::runtime_error);

  // Validation happens before moving, so a rejected spline remains usable.
  EXPECT_EQ(spline.ncoeffs(0), 5);
  EXPECT_EQ(spline.coeffs(0).numel(), 5);
}

} // namespace
