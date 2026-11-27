/**
   @file src/xml_charconv.hpp

   @brief Helpers for the unittests of the XML number formatting and parsing

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <cstring>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <iganet/iganet.h>
#include <iganet/utils/charconv.hpp>

namespace iganet::unittests::xml {

/// @brief Values that exercise the shortest round-trip form: a value with 15
/// significant digits, small values, a subnormal, negative zero, and the
/// largest and smallest normal values of `T`.
template <typename T> std::vector<T> extreme_values() {
  T subnormal;
  if constexpr (sizeof(T) == sizeof(double))
    subnormal = static_cast<T>(1e-310);
  else
    subnormal = static_cast<T>(1e-40f);

  return {static_cast<T>(0.123456789012345),
          static_cast<T>(1e-9),
          static_cast<T>(4.9e-7),
          subnormal,
          static_cast<T>(-0.0),
          std::numeric_limits<T>::max(),
          std::numeric_limits<T>::min()};
}

/// @brief Copies the values of a CPU tensor to a vector.
template <typename T> std::vector<T> to_vector(const torch::Tensor &tensor) {
  auto contiguous = tensor.to(torch::kCPU).contiguous();
  return std::vector<T>(contiguous.template data_ptr<T>(),
                        contiguous.template data_ptr<T>() + contiguous.numel());
}

/// @brief Returns true if both tensors have equal values (`torch::equal`) and
/// identical bit patterns (distinguishes `-0.0` from `0.0`).
template <typename T>
bool bit_equal(const torch::Tensor &a, const torch::Tensor &b) {
  if (!torch::equal(a, b))
    return false;
  const auto va = to_vector<T>(a);
  const auto vb = to_vector<T>(b);
  return va.size() == vb.size() &&
         std::memcmp(va.data(), vb.data(), va.size() * sizeof(T)) == 0;
}

/// @brief Joins values with separators taken in turn from `separators`,
/// framed by `lead` and `trail`.
template <typename T>
std::string join(const std::vector<T> &values,
                 const std::vector<std::string> &separators,
                 const std::string &lead = "", const std::string &trail = "") {
  std::string text = lead;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      text += separators[(i - 1) % separators.size()];
    iganet::utils::append_number(text, values[i]);
  }
  return text + trail;
}

/// @brief Whitespace layouts of a value list: name and separators.
inline const std::vector<std::pair<std::string, std::vector<std::string>>> &
whitespace_forms() {
  static const std::vector<std::pair<std::string, std::vector<std::string>>>
      forms{{"single spaces", {" "}},
            {"one value per line", {"\n"}},
            {"tabs", {"\t"}},
            {"multiple spaces", {"   "}},
            {"carriage returns", {"\r\n"}},
            {"two values per line", {" ", "\n"}},
            {"mixed", {" ", "\t", "\n", "  \t\r\n  "}}};
  return forms;
}

/// @brief Sets the global C and C++ locale to `de_DE.UTF-8` (decimal comma)
/// for its lifetime if that locale is installed.
class DecimalCommaLocale {
public:
  DecimalCommaLocale() {
    try {
      const std::locale german("de_DE.UTF-8");
      if (std::use_facet<std::numpunct<char>>(german).decimal_point() != ',')
        return;
      previous_ = std::locale::global(german);
      active_ = true;
    } catch (const std::runtime_error &) {
    }
  }

  ~DecimalCommaLocale() {
    if (active_)
      std::locale::global(previous_);
  }

  DecimalCommaLocale(const DecimalCommaLocale &) = delete;
  DecimalCommaLocale &operator=(const DecimalCommaLocale &) = delete;

  /// @brief Returns true if the locale was installed and is in effect.
  bool active() const noexcept { return active_; }

private:
  std::locale previous_;
  bool active_ = false;
};

/// @brief Returns the message of the `std::runtime_error` thrown by `f`, or an
/// empty string if `f` does not throw it.
template <typename F> std::string runtime_error_message(F &&f) {
  try {
    f();
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  return "";
}

} // namespace iganet::unittests::xml
