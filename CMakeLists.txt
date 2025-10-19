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

option(ENABLE_TESTS "Build unit tests" OFF)
option(ENABLE_BENCHMARKS "Build benchmarks" OFF)

#
# Popnet Lib
#

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(AVX512_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd -mavx512vbmi -mavx512vnni")
elseif (MSVC)
    set(AVX512_FLAGS "/arch:AVX512")
endif()

file(GLOB_RECURSE POPNET_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(popnet STATIC ${POPNET_SOURCES})

target_include_directories(popnet PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(popnet PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/") 
target_include_directories(popnet PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu")
target_include_directories(popnet PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/x86")