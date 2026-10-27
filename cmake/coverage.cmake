# Requires Clang, matching LLVM tools, and gcovr 8.5+ on PATH.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "POPNET_ENABLE_COVERAGE requires Clang or clang-cl")
endif()
if(NOT POPNET_BUILD_TESTS)
  message(FATAL_ERROR "POPNET_ENABLE_COVERAGE requires POPNET_BUILD_TESTS=ON")
endif()
if(POPNET_LTO OR POPNET_ASAN OR POPNET_UBSAN OR POPNET_TSAN)
  message(FATAL_ERROR "Use a separate coverage build with LTO and sanitizers disabled")
endif()

find_program(GCOVR gcovr REQUIRED)
execute_process(COMMAND "${GCOVR}" --version
  OUTPUT_VARIABLE gcovr_version RESULT_VARIABLE gcovr_result)
string(REGEX MATCH "gcovr ([0-9.]+)" gcovr_version "${gcovr_version}")
if(NOT gcovr_result EQUAL 0 OR NOT gcovr_version OR CMAKE_MATCH_1 VERSION_LESS 8.5)
  message(FATAL_ERROR "Coverage requires gcovr 8.5 or newer")
endif()

get_filename_component(llvm_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
string(REGEX MATCH "^[0-9]+" llvm_major "${CMAKE_CXX_COMPILER_VERSION}")
find_program(LLVM_PROFDATA NAMES llvm-profdata-${llvm_major} llvm-profdata
  HINTS "${llvm_bin}" REQUIRED)
# gcovr derives the llvm-cov path from the llvm-profdata path.
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
    list(APPEND binaries --llvm-cov-binary "$<TARGET_FILE:${target}>")
  endforeach()
  set(profiles "${PROJECT_BINARY_DIR}/profiles")

  add_custom_target(coverage
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${profiles}"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${PROJECT_BINARY_DIR}/coverage.xml" "${PROJECT_BINARY_DIR}/coverage.html"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${profiles}"
    COMMAND "${CMAKE_COMMAND}" -E env "LLVM_PROFILE_FILE=${profiles}/%m.profraw"
      "${CMAKE_CTEST_COMMAND}" --test-dir "${PROJECT_BINARY_DIR}" -C "$<CONFIG>" --output-on-failure --no-tests=error
    COMMAND "${GCOVR}" --root "${PROJECT_SOURCE_DIR}" --filter "src/" --filter "include/"
      --llvm-profdata-executable "${LLVM_PROFDATA}" ${binaries}
      --cobertura "${PROJECT_BINARY_DIR}/coverage.xml"
      --html-details "${PROJECT_BINARY_DIR}/coverage.html" --print-summary "${profiles}"
    DEPENDS ${ARGN}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    USES_TERMINAL VERBATIM)
endfunction()
