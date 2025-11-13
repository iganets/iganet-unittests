########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2025 by the IgANet authors
#
# This file is part of the IgANet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

cmake_minimum_required (VERSION 3.24)

include(FetchContent)

########################################################################
#
# Project: IgANet-UnitTests
#
########################################################################
project(IgANet-UnitTests)

add_library(iganet_unittests INTERFACE)
add_library(iganet::unittests ALIAS iganet_unittests)
target_include_directories(iganet_unittests INTERFACE ${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)

########################################################################
# IgANets (core)
########################################################################
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

  if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    FetchContent_Declare(
      iganet_core
      GIT_REPOSITORY https://github.com/iganets/iganet.git
      GIT_TAG main
    )
  else()  
    FetchContent_Declare(
      iganet_core
      URL https://github.com/iganets/iganet/archive/refs/heads/main.zip
    )
  endif()
     
  FetchContent_MakeAvailable(iganet_core)
  FetchContent_GetProperties(iganet_core)
  
  ######################################################################
  # Set C++ standard
  ######################################################################
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  enable_testing()
endif()

########################################################################
# GoogleTest
########################################################################
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

########################################################################
# BSplineLib
########################################################################
FetchContent_Declare(
  bsplinelib
  URL https://github.com/tataratat/BSplineLib/archive/refs/heads/main.zip
  )
set(BSPLINELIB_SHARED ON CACHE BOOL "" FORCE)
set(BSPLINELIB_BUILD_PUGIXML OFF CACHE BOOL "" FORCE)
set(pugixml_DIR ${pugixml_BINARY_DIR})
FetchContent_MakeAvailable(googletest bsplinelib)
target_link_libraries(iganet_unittests INTERFACE gtest_main splines)
target_compile_definitions(iganet_unittests INTERFACE UNITTEST_DISABLE_SHORT_MACROS)

########################################################################
# Options
########################################################################
# Set a default coefficient numeric types if not specified
if(NOT IGANET_UNITTESTS_COEFF_TYPE)
  set (IGANET_UNITTESTS_COEFF_TYPE "double" CACHE STRING
    "Coefficient type(float, double)" FORCE)
endif()
set_property(CACHE IGANET_UNITTESTS_COEFF_TYPE PROPERTY STRINGS
  "float" "double")

message("")
message("UnitTests options:")
message("IGANET_UNITTESTS_COEFF_TYPE........: ${IGANET_UNITTESTS_COEFF_TYPE}")
message("")

########################################################################
# Config file
########################################################################
configure_file(${PROJECT_SOURCE_DIR}/src/unittest_config.hpp.in
  "${PROJECT_BINARY_DIR}/src/unittest_config.hpp" @ONLY)

########################################################################
# Collect unit test sources
########################################################################
file(GLOB UNITTESTS src/*.cxx)

foreach(unittest ${UNITTESTS})

  get_filename_component(name ${unittest} NAME_WE)
  add_executable(${name} ${unittest})

  if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_compile_options(${name} PRIVATE "-DIGANET_DATA_DIR=\"${PROJECT_SOURCE_DIR}/filedata/\"")
  target_link_libraries(${name} PRIVATE iganet::core)
  target_link_libraries(${name} PRIVATE iganet::unittests)
  
  add_test(${name} ${name})
  
endforeach()
