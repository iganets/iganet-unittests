/**
   @file unittests/src/unittest_getenv.cxx

   @brief Environment variable utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet/iganet.h>
#include <gtest/gtest.h>

TEST(Getenv, DefaultsAndTypedParsing) {
  const char *missing = "IGANET_UNITTEST_MISSING_VARIABLE_7D614D";
  unsetenv(missing);
  EXPECT_EQ(iganet::utils::getenv(missing, 42), 42);
  EXPECT_EQ(iganet::utils::getenv(missing, std::string{"fallback"}), "fallback");
  EXPECT_EQ(iganet::utils::getenv(missing, {1, 2}), (std::vector{1, 2}));

  setenv("IGANET_UNITTEST_INTEGER", "-17", 1);
  setenv("IGANET_UNITTEST_REAL", "2.75", 1);
  setenv("IGANET_UNITTEST_LIST", "1,2,5", 1);
  EXPECT_EQ(iganet::utils::getenv("IGANET_UNITTEST_INTEGER", 0), -17);
  EXPECT_DOUBLE_EQ(iganet::utils::getenv("IGANET_UNITTEST_REAL", 0.0), 2.75);
  EXPECT_EQ(iganet::utils::getenv("IGANET_UNITTEST_LIST", {0, 0}),
            (std::vector{1, 2, 5}));
  unsetenv("IGANET_UNITTEST_INTEGER");
  unsetenv("IGANET_UNITTEST_REAL");
  unsetenv("IGANET_UNITTEST_LIST");
}

TEST(Getenv, ParsesStringAndFloatingPointLists) {
  setenv("IGANET_UNITTEST_STRING_LIST", "cpu,cuda,mps", 1);
  setenv("IGANET_UNITTEST_REAL_LIST", "1.25,-2.5,0", 1);
  EXPECT_EQ(iganet::utils::getenv<std::string>("IGANET_UNITTEST_STRING_LIST", {}),
            (std::vector<std::string>{"cpu", "cuda", "mps"}));
  EXPECT_EQ(iganet::utils::getenv<double>("IGANET_UNITTEST_REAL_LIST", {}),
            (std::vector<double>{1.25, -2.5, 0.0}));
  unsetenv("IGANET_UNITTEST_STRING_LIST");
  unsetenv("IGANET_UNITTEST_REAL_LIST");
}
