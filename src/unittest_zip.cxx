/**
   @file unittests/unittest_zip.cxx

   @brief Zip utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet/iganet.h>
#include <gtest/gtest.h>

TEST(Zip, IteratesInLockstepAndExposesReferences) {
  std::vector<int> numbers{1, 2, 3};
  std::array<std::string, 3> words{"one", "two", "three"};
  int expected = 1;
  for (auto [number, word] : iganet::utils::zip(numbers, words)) {
    EXPECT_EQ(number, expected++);
    number *= 10;
    EXPECT_FALSE(word.empty());
  }
  EXPECT_EQ(numbers, (std::vector{10, 20, 30}));
}

TEST(Zip, SupportsPostIncrement) {
  std::vector<int> a{1, 2};
  std::vector<int> b{3, 4};
  auto zipped = iganet::utils::zip(a, b);
  auto iterator = zipped.begin();
  auto previous = iterator++;
  EXPECT_EQ(std::get<0>(*previous), 1);
  EXPECT_EQ(std::get<0>(*iterator), 2);
}

TEST(Zip, SupportsConstThreeWayAndEmptySequences) {
  const std::vector<int> a{1, 2};
  const std::array<char, 2> b{'a', 'b'};
  const std::vector<double> c{1.5, 2.5};
  int count = 0;
  for (auto [number, letter, real] : iganet::utils::zip(a, b, c)) {
    EXPECT_EQ(number, count + 1);
    EXPECT_EQ(letter, count == 0 ? 'a' : 'b');
    EXPECT_DOUBLE_EQ(real, count + 1.5);
    ++count;
  }
  EXPECT_EQ(count, 2);

  std::vector<int> empty;
  std::vector<int> also_empty;
  EXPECT_EQ(iganet::utils::zip(empty, also_empty).begin() !=
                iganet::utils::zip(empty, also_empty).end(),
            false);
}

TEST(Zip, RejectsUnequalLengthRanges) {
  std::vector<int> short_range{1, 2};
  std::vector<int> long_range{3, 4, 5};
  EXPECT_THROW((void)iganet::utils::zip(short_range, long_range),
               std::invalid_argument);

  std::vector<int> empty;
  EXPECT_THROW((void)iganet::utils::zip(empty, short_range),
               std::invalid_argument);
}
