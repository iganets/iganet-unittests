/**
   @file unittests/src/unittest_memory.cxx

   @brief Memory debugger unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet/iganet.h>
#include <gtest/gtest.h>

#include <sstream>

TEST(MemoryDebugger, TracksGenericTensorAndArrayObjects) {
  iganet::MemoryDebugger<42> debugger;
  int scalar = 7;
  debugger.add("scalar", scalar);
  debugger.add("tensor", torch::ones({8}, torch::kFloat64));
  debugger.add("array", std::array<int, 2>{1, 2});

  std::ostringstream stream;
  stream << debugger;
  EXPECT_NE(stream.str().find("Memory debugger (ID=42)"), std::string::npos);
  EXPECT_NE(stream.str().find("scalar"), std::string::npos);
  EXPECT_NE(stream.str().find("tensor"), std::string::npos);
  EXPECT_NE(stream.str().find("array0"), std::string::npos);
  EXPECT_NE(stream.str().find("[     Total  ]"), std::string::npos);
}

TEST(MemoryDebugger, ClearResetsCountersAndEntries) {
  iganet::MemoryDebugger<7> debugger;
  debugger.add("value", 1.0);
  debugger.clear();
  std::ostringstream stream;
  debugger.pretty_print(stream);
  EXPECT_EQ(stream.str().find("value"), std::string::npos);
  EXPECT_NE(stream.str().find("[     Total  ]"), std::string::npos);
  EXPECT_NE(stream.str().find("0b"), std::string::npos);
}
