/**
   @file unittests/src/unittest_integer_pow.cxx

   @brief Integer power utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

TEST(IntegerPow, HandlesZeroAndPositiveExponentsAtCompileTime) {
  static_assert(iganet::utils::integer_pow<2, 0>::value == 1);
  static_assert(iganet::utils::integer_pow<2, 10>::value == 1024);
  static_assert(iganet::utils::integer_pow<-3, 3>::value == -27);
  EXPECT_EQ((iganet::utils::integer_pow<5, 4>::value), 625);
}
