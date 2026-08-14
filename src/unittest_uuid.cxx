/**
   @file unittests/src/unittest_uuid.cxx

   @brief UUID utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>
#include <regex>
#include <set>

TEST(Uuid, HasExpectedFormatAndProducesDistinctValues) {
  const std::regex pattern("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$");
  std::set<std::string> values;
  for (int i = 0; i < 32; ++i) {
    auto value = iganet::utils::uuid::create();
    EXPECT_TRUE(std::regex_match(value, pattern));
    values.insert(std::move(value));
  }
  EXPECT_EQ(values.size(), 32);
}
