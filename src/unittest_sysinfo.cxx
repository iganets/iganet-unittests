/**
   @file unittests/unittest_sysinfo.cxx

   @brief System information unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

TEST(SysInfo, ReportsNonemptyBuildAndRuntimeInformation) {
  EXPECT_FALSE(iganet::getIgANetVersion().empty());
  EXPECT_FALSE(iganet::getCompilerVersion().empty());
  EXPECT_FALSE(iganet::getCppVersion().empty());
  EXPECT_FALSE(iganet::getStdLibVersion().empty());
  EXPECT_FALSE(iganet::getExtraLibsVersion().empty());
  EXPECT_FALSE(iganet::getCpuInfo().empty());
  EXPECT_FALSE(iganet::getMemoryInfo().empty());

  auto version = iganet::getVersion();
  EXPECT_NE(version.find("IgANets - Isogeometric Analysis Networks"),
            std::string::npos);
  EXPECT_NE(version.find(iganet::getIgANetVersion()), std::string::npos);
  EXPECT_NE(version.find("#intraop threads"), std::string::npos);
}

TEST(SysInfo, ExecutablePathExistsAndIsADirectory) {
  auto path = iganet::getExecutablePath();
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_TRUE(std::filesystem::is_directory(path));
}
