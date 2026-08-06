/**
   @file unittests/unittest_index_sequence.cxx

   @brief Index sequence utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>
#include <type_traits>

TEST(IndexSequence, ReversesCompileTimeIndices) {
  static_assert(std::is_base_of_v<std::index_sequence<0>,
                                  iganet::utils::make_reverse_index_sequence<1>>);
  static_assert(std::is_base_of_v<
                std::index_sequence<4, 3, 2, 1, 0>,
                iganet::utils::make_reverse_index_sequence<5>>);
  SUCCEED();
}
