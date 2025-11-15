#
# Google Benchmark
#

include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
)

FetchContent_MakeAvailable(googlebenchmark)

#
# Benchmark Executable
#

file(GLOB_RECURSE POPNET_BENCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(popnet_bench ${POPNET_BENCH_SOURCES})

target_include_directories(popnet_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(popnet_bench PRIVATE popnet_internal benchmark::benchmark benchmark::benchmark_main)