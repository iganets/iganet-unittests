/**
   @file unittests/unittest_fqn.cxx

   @brief Full qualified name utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>
#include <sstream>

namespace {
class Named final : public iganet::utils::FullQualifiedName {
public:
  void pretty_print(std::ostream &os) const noexcept override { os << name(); }
};
}

TEST(FullQualifiedName, ReportsDynamicTypeAndPrettyPrints) {
  Named object;
  EXPECT_NE(object.name().find("Named"), std::string::npos);
  EXPECT_EQ(&object.name(), &object.name());
  std::ostringstream stream;
  object.pretty_print(stream);
  EXPECT_EQ(stream.str(), object.name());
}

TEST(FullQualifiedName, WorksThroughBaseReference) {
  Named object;
  const iganet::utils::FullQualifiedName &base = object;
  EXPECT_NE(base.name().find("Named"), std::string::npos);
  std::ostringstream stream;
  base.pretty_print(stream);
  EXPECT_EQ(stream.str(), base.name());
}
