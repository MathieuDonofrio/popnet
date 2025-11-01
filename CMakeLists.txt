cmake_minimum_required(VERSION 3.20)

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

# Build Artefacts

option(POPNET_BUILD_TESTS "Build unit tests" OFF)
option(POPNET_BUILD_BENCHMARKS "Build performance benchmarks" OFF)

# Optimization and Compilation

option(POPNET_NATIVE "Enable native CPU tuning when building on the host system. Automatically disabled when cross-compiling." ON)
option(POPNET_LTO "Enable link-time optimization (LTO) for improved runtime performance. Requires compiler and linker support." OFF)

# Compute Backend

set(POPNET_BACKEND "CPU" CACHE STRING "Select the compute backend (CPU, CUDA)")
set_property(CACHE POPNET_BACKEND PROPERTY STRINGS CPU CUDA)

# Sanitizers

option(POPNET_ASAN "Enable AddressSanitizer (ASAN) for detecting memory errors" OFF)
option(POPNET_UBSAN "Enable UndefinedBehaviorSanitizer (UBSAN) for detecting undefined behavior" OFF)
option(POPNET_TSAN "Enable ThreadSanitizer (TSAN) for detecting data races" OFF)

#
# Includes
#

include(cmake/cpu-native-tuning.cmake)
include(cmake/interprocedural-optimization.cmake)
include(cmake/sanitizers.cmake)

#
# Popnet Lib
#

if(POPNET_BACKEND STREQUAL "CPU")
    set(POPNET_COMPUTE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/compute/cpu")
elseif(POPNET_BACKEND STREQUAL "CUDA")
    set(POPNET_COMPUTE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/compute/cuda")
else()
    message(FATAL_ERROR "Unsupported POPNET_BACKEND: ${POPNET_BACKEND}")
endif()

file(GLOB POPNET_COMMON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB POPNET_COMMON_COMPUTE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/compute/*.cpp")
file(GLOB POPNET_COMPUTE_SOURCES CONFIGURE_DEPENDS "${POPNET_COMPUTE_DIR}/*.cpp")

add_library(popnet STATIC ${POPNET_COMMON_SOURCES} ${POPNET_COMMON_COMPUTE_SOURCES} ${POPNET_COMPUTE_SOURCES})

target_include_directories(popnet
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/compute"
            "${POPNET_COMPUTE_DIR}"
)

target_include_directories(popnet PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

enable_popnet_ipo(popnet) 

add_library(popnet_internal INTERFACE)

target_include_directories(popnet_internal INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compute"
    "${POPNET_COMPUTE_DIR}"
)

target_link_libraries(popnet_internal INTERFACE popnet)

#
# Testing
#

if(POPNET_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
