/**
   @file unittests/src/unittest_tuple.cxx

   @brief Tuple utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet/iganet.h>
#include <gtest/gtest.h>
#include <type_traits>

TEST(Tuple, TraitsConcatenationAndRepeat) {
  static_assert(iganet::utils::is_tuple_v<std::tuple<int, double>>);
  static_assert(!iganet::utils::is_tuple_v<int>);
  static_assert(!iganet::utils::is_tuple_of_tuples_v<int>);
  static_assert(iganet::utils::is_tuple_of_tuples_v<
                std::tuple<std::tuple<int>, std::tuple<double>>>);
  static_assert(!iganet::utils::is_tuple_of_tuples_v<
                std::tuple<std::tuple<int>, double>>);
  static_assert(iganet::utils::is_tuple_of_tuples_v<std::tuple<>>);
  static_assert(std::is_same_v<iganet::utils::tuple_cat_t<
      std::tuple<int>, double, std::tuple<char>>, std::tuple<int, double, char>>);
  EXPECT_EQ(iganet::utils::repeat_tuple<3>(7), std::make_tuple(7, 7, 7));
}

TEST(Tuple, ConcatenatesAndTransformsTensors) {
  auto tuple = std::make_tuple(torch::tensor({1., 2.}), torch::tensor({3.}));
  EXPECT_TRUE(torch::equal(iganet::utils::cat_tuple_into_tensor(tuple),
                           torch::tensor({1., 2., 3.})));
  EXPECT_TRUE(torch::equal(
      iganet::utils::cat_tuple_into_tensor(tuple, [](auto t) { return 2 * t; }),
      torch::tensor({2., 4., 6.})));
}

TEST(Tuple, SlicesTensorIntoTuple) {
  auto tuple = std::make_tuple(torch::zeros(2), torch::zeros(1));
  iganet::utils::slice_tensor_into_tuple(
      tuple, torch::tensor({1., 2., 3.}),
      [](const auto &t) { return t.numel(); },
      [](auto &target, const auto &slice) { target = slice.clone(); });
  EXPECT_TRUE(torch::equal(std::get<0>(tuple), torch::tensor({1., 2.})));
  EXPECT_TRUE(torch::equal(std::get<1>(tuple), torch::tensor({3.})));
}

TEST(Tuple, EmptyConcatenationAndExplicitOffsetSlicing) {
  static_assert(std::is_same_v<iganet::utils::tuple_cat_t<>, std::tuple<>>);
  auto tuple = std::make_tuple(torch::zeros(1), torch::zeros(2));
  int64_t offset = 1;
  iganet::utils::slice_tensor_into_tuple(
      tuple, torch::tensor({9., 1., 2., 3.}),
      [](const auto &t) { return t.numel(); },
      [](auto &target, const auto &slice) { target = slice.clone(); }, offset,
      0);
  EXPECT_EQ(offset, 4);
  EXPECT_TRUE(torch::equal(std::get<0>(tuple), torch::tensor({1.})));
  EXPECT_TRUE(torch::equal(std::get<1>(tuple), torch::tensor({2., 3.})));
}

TEST(Tuple, ConcatenatesAndSlicesAlongNonzeroDimension) {
  auto tuple = std::make_tuple(torch::zeros({2, 1}), torch::zeros({2, 2}));
  auto source = torch::tensor({{1., 2., 3.}, {4., 5., 6.}});
  iganet::utils::slice_tensor_into_tuple(
      tuple, source, [](const auto &t) { return t.size(1); },
      [](auto &target, const auto &slice) { target = slice.clone(); }, 1);
  EXPECT_TRUE(torch::equal(std::get<0>(tuple),
                           torch::tensor({{1.}, {4.}})));
  EXPECT_TRUE(torch::equal(std::get<1>(tuple),
                           torch::tensor({{2., 3.}, {5., 6.}})));
  EXPECT_TRUE(torch::equal(iganet::utils::cat_tuple_into_tensor(tuple, 1),
                           source));
}
