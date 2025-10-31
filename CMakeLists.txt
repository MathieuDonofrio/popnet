cmake_minimum_required(VERSION 3.16)

#
# Project
#

project(Popnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
 
#
# Options
#

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

#
# Popnet Lib
#

file(GLOB_RECURSE POPNET_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(popnet STATIC ${POPNET_SOURCES})

target_include_directories(popnet
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/x86"
)

target_include_directories(popnet PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(popnet_internal INTERFACE)

target_include_directories(popnet_internal INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/x86"
)

target_link_libraries(popnet_internal INTERFACE popnet)

#
# Testing
#

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
