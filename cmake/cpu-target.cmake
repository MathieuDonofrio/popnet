include(CheckCXXSourceCompiles)

function(popnet_get_cpu_target flag_out)
  set(${flag_out} "" PARENT_SCOPE)
  string(TOUPPER "${POPNET_CPU_TARGET}" mode)
  if(mode STREQUAL "DEFAULT")
    return()
  endif()

  if(NOT POPNET_CPU_TARGET MATCHES "^[A-Za-z0-9_][A-Za-z0-9_.+-]*$" OR mode MATCHES "^NATIVE[+]")
    message(FATAL_ERROR "Invalid POPNET_CPU_TARGET='${POPNET_CPU_TARGET}'. Use NATIVE or a single explicit target name.")
  endif()

  set(cpu "${POPNET_CPU_TARGET}")
  if(mode STREQUAL "NATIVE")
    # Some same-OS cross builds do not set CMAKE_CROSSCOMPILING.
    string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" host_arch)
    string(TOLOWER "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" compiler_arch)
    string(REGEX REPLACE "^(amd64|x86_64)$" "x64" host_arch "${host_arch}")
    if(CMAKE_CROSSCOMPILING
       OR (APPLE AND CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
       OR (WIN32 AND compiler_arch AND NOT compiler_arch STREQUAL host_arch))
      message(FATAL_ERROR "POPNET_CPU_TARGET=NATIVE is unavailable when cross-compiling. Use DEFAULT or an explicit target.")
    endif()
    set(cpu native)
  endif()

  set(flags "")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(mode STREQUAL "NATIVE")
      popnet_msvc_native_flags(flags)
    else()
      set(flags "/arch:${cpu}")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set(flags "-march=${cpu}" "-mcpu=${cpu}")
    if(mode STREQUAL "NATIVE")
      list(REVERSE flags)
    endif()
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      list(TRANSFORM flags PREPEND "/clang:")
      if(NOT mode STREQUAL "NATIVE")
        list(APPEND flags "/arch:${cpu}")
      endif()
      string(APPEND CMAKE_REQUIRED_FLAGS " /clang:-Werror")
    else()
      string(APPEND CMAKE_REQUIRED_FLAGS " -Werror")
    endif()
  endif()

  # Compile only. Some drivers warn and ignore an option or accept a tuning-only alias.
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(required_flags "${CMAKE_REQUIRED_FLAGS}")
  foreach(flag IN LISTS flags)
    set(CMAKE_REQUIRED_FLAGS "${required_flags} ${flag}")
    unset(POPNET_CPU_TARGET_FLAG_SUPPORTED CACHE)
    check_cxx_source_compiles("int main() { return 0; }" POPNET_CPU_TARGET_FLAG_SUPPORTED
      FAIL_REGEX "D9002" "[-]Wunused-command-line-argument" "-mcpu=.*-mtune=.*-march=")
    if(POPNET_CPU_TARGET_FLAG_SUPPORTED)
      unset(POPNET_CPU_TARGET_FLAG_SUPPORTED CACHE)
      message(STATUS "Popnet CPU target ${POPNET_CPU_TARGET}: ${flag}")
      set(${flag_out} "${flag}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  unset(POPNET_CPU_TARGET_FLAG_SUPPORTED CACHE)

  if(mode STREQUAL "NATIVE" AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(STATUS "Popnet CPU target NATIVE: using MSVC defaults")
    return()
  endif()

  string(JOIN " " attempted_flags ${flags})
  message(FATAL_ERROR
    "Unsupported POPNET_CPU_TARGET='${POPNET_CPU_TARGET}' for ${CMAKE_CXX_COMPILER_ID} "
    "${CMAKE_CXX_COMPILER_VERSION}. Tried: ${attempted_flags}")
endfunction()

function(popnet_configure_cpu_target target scope)
  target_compile_options(${target} ${scope} ${popnet_cpu_target_flags})
  if(NOT POPNET_ENABLE_SIMD)
    target_compile_definitions(${target} ${scope} POPNET_DISABLE_SIMD=1)
  endif()
endfunction()

function(popnet_msvc_native_flags flags_out)
  if(NOT CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(X86|x64)$")
    return()
  endif()

  set(probe "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/popnet-msvc-native.cpp")
  file(WRITE "${probe}" [=[
#include <cstdio>
#include <intrin.h>
#include <windows.h>

int main()
{
  constexpr auto avx_state = XSTATE_MASK_LEGACY_SSE | XSTATE_MASK_AVX;
  const auto state = GetEnabledXStateFeatures();
  if ((state & avx_state) != avx_state) return 0;

  int cpu[4];
  __cpuid(cpu, 0);
  const int max_leaf = cpu[0];
  __cpuid(cpu, 1);
  const auto features = static_cast<unsigned>(cpu[2]);
  if (!(features & (1u << 28))) return 0; // AVX

  std::puts("/arch:AVX");

  // MSVC /arch:AVX2 permits instructions beyond AVX2.
  constexpr unsigned fma_movbe_f16c = (1u << 12) | (1u << 22) | (1u << 29);
  if (max_leaf < 7 || (features & fma_movbe_f16c) != fma_movbe_f16c) return 0;

  __cpuidex(cpu, 7, 0);
  const auto extended = static_cast<unsigned>(cpu[1]);
  constexpr unsigned bmi_avx2_bmi2 = (1u << 3) | (1u << 5) | (1u << 8);
  if ((extended & bmi_avx2_bmi2) != bmi_avx2_bmi2) return 0;

  __cpuid(cpu, 0x80000000);
  if (static_cast<unsigned>(cpu[0]) < 0x80000001u) return 0;
  __cpuid(cpu, 0x80000001);
  if (!(cpu[2] & (1u << 5))) return 0; // LZCNT

  std::puts("/arch:AVX2");

  // /arch:AVX512 also permits DQ, CD, BW and VL.
  constexpr unsigned avx512 = (1u << 16) | (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);
  if ((extended & avx512) == avx512 && (state & XSTATE_MASK_AVX512) == XSTATE_MASK_AVX512)
    std::puts("/arch:AVX512");
}
]=])

  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  try_run(popnet_msvc_run popnet_msvc_compiled "${CMAKE_CURRENT_BINARY_DIR}" "${probe}" RUN_OUTPUT_VARIABLE output)
  if(popnet_msvc_compiled AND popnet_msvc_run STREQUAL "0")
    string(REGEX MATCHALL "/arch:AVX[0-9]*" flags "${output}")
    list(REVERSE flags)
    set(${flags_out} "${flags}" PARENT_SCOPE)
  endif()
  unset(popnet_msvc_run CACHE)
  unset(popnet_msvc_compiled CACHE)
endfunction()

popnet_get_cpu_target(popnet_cpu_target_flags)
