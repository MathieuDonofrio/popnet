#
# Doctest
#

include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12
)

FetchContent_MakeAvailable(doctest)

#
# Test Executable
#

file(GLOB_RECURSE POPNET_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(popnet_test ${POPNET_TEST_SOURCES})

target_include_directories(popnet_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(popnet_test PRIVATE popnet_internal doctest)

if(POPNET_ENABLE_COVERAGE)
  enable_coverage_for(popnet_test)
  add_coverage_target(popnet_test)
endif()

#
# Test Discovery
#

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

doctest_discover_tests(popnet_test)