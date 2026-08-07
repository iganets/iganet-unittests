/**
   @file unittests/unittest_optimizer.cxx

   @brief Optimizer utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

#include <sstream>
#include <type_traits>

static_assert(iganet::OptimizerType<torch::optim::Adam>);
static_assert(!iganet::OptimizerType<int>);
static_assert(std::is_same_v<iganet::optimizer_options_type<
                                 torch::optim::Adagrad>::type,
                             torch::optim::AdagradOptions>);
static_assert(std::is_same_v<iganet::optimizer_options_type<
                                 torch::optim::Adam>::type,
                             torch::optim::AdamOptions>);
static_assert(std::is_same_v<iganet::optimizer_options_type<
                                 torch::optim::AdamW>::type,
                             torch::optim::AdamWOptions>);
static_assert(std::is_same_v<iganet::optimizer_options_type<
                                 torch::optim::LBFGS>::type,
                             torch::optim::LBFGSOptions>);
static_assert(std::is_same_v<iganet::optimizer_options_type<
                                 torch::optim::RMSprop>::type,
                             torch::optim::RMSpropOptions>);
static_assert(std::is_same_v<iganet::optimizer_options_type<
                                 torch::optim::SGD>::type,
                             torch::optim::SGDOptions>);

TEST(Optimizer, FormatsAllSupportedOptionTypes) {
  std::ostringstream stream;
  stream << torch::optim::AdagradOptions(0.1)
         << torch::optim::AdamOptions(0.2)
         << torch::optim::AdamWOptions(0.3)
         << torch::optim::LBFGSOptions(0.4)
         << torch::optim::RMSpropOptions(0.5)
         << torch::optim::SGDOptions(0.6);
  auto output = stream.str();
  EXPECT_NE(output.find("AdagradOptions"), std::string::npos);
  EXPECT_NE(output.find("AdamOptions"), std::string::npos);
  EXPECT_NE(output.find("AdamWOptions"), std::string::npos);
  EXPECT_NE(output.find("LBFGSOptions"), std::string::npos);
  EXPECT_NE(output.find("RMSpropOptions"), std::string::npos);
  EXPECT_NE(output.find("SGDOptions"), std::string::npos);
  EXPECT_NE(output.find("lr = 0.6"), std::string::npos);
}
