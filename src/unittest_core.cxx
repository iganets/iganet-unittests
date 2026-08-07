/**
   @file unittests/unittest_core.cxx

   @brief Core utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

#include <sstream>

TEST(Core, IntegerLiteralsHaveExpectedTypesAndValues) {
  using namespace iganet::literals;
  static_assert(std::is_same_v<decltype(1_i8), int8_t>);
  static_assert(std::is_same_v<decltype(1_i16), int16_t>);
  static_assert(std::is_same_v<decltype(1_i32), int32_t>);
  static_assert(std::is_same_v<decltype(1_i64), int64_t>);
  EXPECT_EQ(127_i8, 127);
  EXPECT_EQ(1024_i16, 1024);
  EXPECT_EQ(65536_i32, 65536);
  EXPECT_EQ(1_i64, 1);
}

TEST(Core, VerboseManipulatorStoresStatePerStream) {
  std::ostringstream first;
  std::ostringstream second;
  EXPECT_FALSE(iganet::is_verbose(first));
  first << iganet::verbose;
  EXPECT_TRUE(iganet::is_verbose(first));
  EXPECT_FALSE(iganet::is_verbose(second));
  first << iganet::regular;
  EXPECT_FALSE(iganet::is_verbose(first));
}

TEST(Core, ArrayAndTupleFormattingIsStable) {
  std::ostringstream stream;
  stream << std::array<int, 3>{1, 2, 3} << " "
         << std::tuple<int, std::string>{4, "five"};
  EXPECT_NE(stream.str().find("1"), std::string::npos);
  EXPECT_NE(stream.str().find("five"), std::string::npos);
}

TEST(Core, NullOutputStreamAcceptsWrites) {
  iganet::logging::NullOStream stream;
  EXPECT_NO_THROW(stream << "discarded" << 42 << std::flush);
}
