/**
   @file unittests/unittest_creator.cxx

   @brief Geometry creator unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

#include <sstream>

TEST(Creator, IntervalMapsControlPointsIntoRequestedBounds) {
  iganet::UniformBSpline<double, 1, 1> spline({2});
  iganet::IntervalCreator<double> creator(2.0, 2.0, 5.0, 5.0);
  EXPECT_EQ(&creator.next(spline), &spline);
  const auto coeffs = spline.coeffs(0);
EXPECT_TRUE(torch::allclose(
    coeffs,
    torch::tensor({2., 5.}, coeffs.options())));
  std::ostringstream stream;
  stream << creator;
  EXPECT_NE(stream.str().find("x0min = 2"), std::string::npos);
}

TEST(Creator, RectangleMapsEachCoordinateIndependently) {
  iganet::UniformBSpline<double, 2, 1, 1> spline({2, 2});
  iganet::RectangleCreator<double> creator(1., 1., 3., 3., -2., -2., 4., 4.);
  creator.next(spline);
  EXPECT_DOUBLE_EQ(spline.coeffs(0).min().item<double>(), 1.0);
  EXPECT_DOUBLE_EQ(spline.coeffs(0).max().item<double>(), 3.0);
  EXPECT_DOUBLE_EQ(spline.coeffs(1).min().item<double>(), -2.0);
  EXPECT_DOUBLE_EQ(spline.coeffs(1).max().item<double>(), 4.0);
}

TEST(Creator, CuboidMapsEachCoordinateIndependently) {
  iganet::UniformBSpline<double, 3, 1, 1, 1> spline({2, 2, 2});
  iganet::CuboidCreator<double> creator(1., 1., 2., 2., 3., 3., 4., 4.,
                                        5., 5., 6., 6.);
  creator.next(spline);
  EXPECT_DOUBLE_EQ(spline.coeffs(0).min().item<double>(), 1.0);
  EXPECT_DOUBLE_EQ(spline.coeffs(0).max().item<double>(), 2.0);
  EXPECT_DOUBLE_EQ(spline.coeffs(1).min().item<double>(), 3.0);
  EXPECT_DOUBLE_EQ(spline.coeffs(1).max().item<double>(), 4.0);
  EXPECT_DOUBLE_EQ(spline.coeffs(2).min().item<double>(), 5.0);
  EXPECT_DOUBLE_EQ(spline.coeffs(2).max().item<double>(), 6.0);
}
