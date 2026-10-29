# Requires Clang and matching LLVM coverage tools.
# The coverage target runs this file in script mode after collecting profiles.
if(CMAKE_SCRIPT_MODE_FILE)
  file(GLOB profiles "${PROFILES}/*.profraw")
  if(NOT profiles)
    message(FATAL_ERROR "No coverage profiles were produced")
  endif()
  file(MAKE_DIRECTORY "${OUTPUT_DIR}")
  execute_process(COMMAND "${LLVM_PROFDATA}" merge -sparse ${profiles}
    -o "${OUTPUT_DIR}/coverage.profdata" COMMAND_ERROR_IS_FATAL ANY)

  list(POP_FRONT BINARIES binary)
  set(objects)
  foreach(object IN LISTS BINARIES)
    list(APPEND objects "-object=${object}")
  endforeach()
  set(arguments "${binary}" ${objects} "-instr-profile=${OUTPUT_DIR}/coverage.profdata"
    "${SOURCE_DIR}/src" "${SOURCE_DIR}/include")

  execute_process(COMMAND "${LLVM_COV}" show ${arguments} -format=html
    "-output-dir=${OUTPUT_DIR}" -show-line-counts-or-regions -show-instantiations -show-branches=count
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${LLVM_COV}" export ${arguments} -format=lcov
    OUTPUT_FILE "${OUTPUT_DIR}/coverage.info" COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${LLVM_COV}" report ${arguments}
    COMMAND_ERROR_IS_FATAL ANY)
  return()
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "POPNET_ENABLE_COVERAGE requires Clang or clang-cl")
endif()
if(NOT POPNET_BUILD_TESTS)
  message(FATAL_ERROR "POPNET_ENABLE_COVERAGE requires POPNET_BUILD_TESTS=ON")
endif()
if(POPNET_LTO OR POPNET_ASAN OR POPNET_UBSAN OR POPNET_TSAN)
  message(FATAL_ERROR "Use a separate coverage build with LTO and sanitizers disabled")
endif()

get_filename_component(llvm_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
string(REGEX MATCH "^[0-9]+" llvm_major "${CMAKE_CXX_COMPILER_VERSION}")
find_program(LLVM_PROFDATA NAMES llvm-profdata-${llvm_major} llvm-profdata
  HINTS "${llvm_bin}" REQUIRED)
# Use the coverage reader from the same LLVM installation.
string(REPLACE "llvm-profdata" "llvm-cov" LLVM_COV "${LLVM_PROFDATA}")
execute_process(COMMAND "${LLVM_COV}" --version
  OUTPUT_VARIABLE llvm_version RESULT_VARIABLE llvm_result)
string(REGEX MATCH "LLVM version ([0-9]+)" llvm_version "${llvm_version}")
if(NOT llvm_result EQUAL 0 OR NOT llvm_version OR NOT CMAKE_MATCH_1 STREQUAL llvm_major)
  message(FATAL_ERROR "Coverage requires matching LLVM ${llvm_major} tools; check ${LLVM_PROFDATA}")
endif()

function(enable_coverage_for target)
  target_compile_options(${target} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  get_target_property(type ${target} TYPE)
  if(type STREQUAL "STATIC_LIBRARY")
    set(scope INTERFACE)
  else()
    set(scope PRIVATE)
  endif()

  if(MSVC)
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" /clang:-print-resource-dir
      OUTPUT_VARIABLE clang_resource OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    # clang-cl objects auto-link the runtime; help CMake's direct linker find it.
    target_link_directories(${target} ${scope} "${clang_resource}/lib/windows")
  else()
    target_link_options(${target} ${scope} -fprofile-instr-generate)
  endif()
endfunction()

# Supply test executables and instrumented shared libraries, not static libraries.
function(add_coverage_target)
  set(binaries)
  foreach(target IN LISTS ARGN)
    list(APPEND binaries "$<TARGET_FILE:${target}>")
  endforeach()
  set(profiles "${PROJECT_BINARY_DIR}/profiles")

  add_custom_target(coverage
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${profiles}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${PROJECT_BINARY_DIR}/coverage"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${PROJECT_BINARY_DIR}/coverage.xml" "${PROJECT_BINARY_DIR}/coverage.html"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${profiles}"
    COMMAND "${CMAKE_COMMAND}" -E env "LLVM_PROFILE_FILE=${profiles}/%m.profraw"
      "${CMAKE_CTEST_COMMAND}" --test-dir "${PROJECT_BINARY_DIR}" -C "$<CONFIG>" --output-on-failure --no-tests=error
    COMMAND "${CMAKE_COMMAND}" "-DLLVM_PROFDATA=${LLVM_PROFDATA}" "-DLLVM_COV=${LLVM_COV}"
      "-DBINARIES=${binaries}" "-DPROFILES=${profiles}"
      "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" "-DOUTPUT_DIR=${PROJECT_BINARY_DIR}/coverage"
      -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
    DEPENDS ${ARGN}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    USES_TERMINAL VERBATIM)
endfunction()
