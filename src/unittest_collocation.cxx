/**
   @file unittests/src/unittest_collocation.cxx

   @brief Collocation point selection unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

TEST(Collocation, SelectsGrevilleInteriorAndRefinedPointSets) {
  using BSpline = iganet::UniformBSpline<double, 1, 2>;
  using Space = iganet::S<BSpline>;
  Space space({5});

  auto all = iganet::CollPtsHelper<Space>::collPts(
      iganet::collPts::greville, space);
  auto interior = iganet::CollPtsHelper<Space>::collPts(
      iganet::collPts::greville_interior, space);
  auto refined1 = iganet::CollPtsHelper<Space>::collPts(
      iganet::collPts::greville_ref1, space);
  auto refined2 = iganet::CollPtsHelper<Space>::collPts(
      iganet::collPts::greville_ref2, space);
  auto refined3 = iganet::CollPtsHelper<Space>::collPts(
      iganet::collPts::greville_ref3, space);

  EXPECT_GT(all.first[0].numel(), interior.first[0].numel());
  EXPECT_GT(refined1.first[0].numel(), all.first[0].numel());
  EXPECT_GT(refined2.first[0].numel(), refined1.first[0].numel());
  EXPECT_GT(refined3.first[0].numel(), refined2.first[0].numel());
  EXPECT_EQ(std::get<0>(all.second).numel() +
                std::get<1>(all.second).numel(),
            2);
}

TEST(Collocation, CoversInteriorRefinementVariantsAndInvalidSpecifier) {
  using Space = iganet::S<iganet::UniformBSpline<double, 1, 2>>;
  Space space({5});
  for (auto kind : {iganet::collPts::greville_interior_ref1,
                    iganet::collPts::greville_interior_ref2,
                    iganet::collPts::greville_interior_ref3}) {
    auto points = iganet::CollPtsHelper<Space>::collPts(kind, space);
    EXPECT_GT(points.first[0].numel(), 0);
    EXPECT_EQ(std::get<0>(points.second).numel() +
                  std::get<1>(points.second).numel(),
              2);
  }
  EXPECT_THROW((void)iganet::CollPtsHelper<Space>::collPts(
                   static_cast<iganet::collPts>(99), space),
               std::runtime_error);
}
