/**
   @file src/common.hpp

   @brief IGANets common helper functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <iganet.h>

namespace iganet {
  std::filesystem::path getDataPath(std::string name="")
  {
    auto exePath = getExecutablePath();
    return std::filesystem::canonical(exePath / "../share/iganet/unittests/filedata/" / name);
  }
} // namespace iganet
