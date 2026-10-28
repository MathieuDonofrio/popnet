if(POPNET_NATIVE)
  if(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling: ignoring POPNET_CPU_NATIVE=ON.")
    set(POPNET_CPU_NATIVE OFF CACHE BOOL "" FORCE)
  else()
    # Autodetect CPU flags

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
      if(HAS_MARCH_NATIVE)
        message(STATUS "Using -march=native -mtune=native for ${CMAKE_CXX_COMPILER_ID}.")
        add_compile_options(-march=native -mtune=native)
      else()
        message(WARNING "${CMAKE_CXX_COMPILER_ID} does not support -march=native.")
      endif()
    elseif(MSVC)
      include(CheckCXXCompilerFlag)

      include(CheckCXXSourceRuns)
      set(NATIVE_FLAGS "")

      # Compiler support alone does not mean the host can execute the ISA.
      if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(x64|X86)$")
        foreach(isa IN ITEMS AVX512 AVX2 AVX)
          check_cxx_compiler_flag("/arch:${isa}" HAS_ARCH_${isa})
          if(HAS_ARCH_${isa})
            set(feature "PF_${isa}_INSTRUCTIONS_AVAILABLE")
            set(extra_check "")
            if(isa STREQUAL "AVX512")
              set(feature PF_AVX512F_INSTRUCTIONS_AVAILABLE)
              # /arch:AVX512 also permits DQ, CD, BW and VL instructions.
              set(extra_check "int cpu[4]; __cpuidex(cpu, 7, 0);
                if ((static_cast<unsigned>(cpu[1]) & 0xd0030000u) != 0xd0030000u) return 1;")
            endif()
            check_cxx_source_runs("#include <windows.h>
              #include <intrin.h>
              int main() {
                ${extra_check}
                return IsProcessorFeaturePresent(${feature}) ? 0 : 1;
              }" HOST_SUPPORTS_${isa})
            if(HOST_SUPPORTS_${isa})
              set(NATIVE_FLAGS "/arch:${isa}")
              break()
            endif()
          endif()
        endforeach()
      endif()

      message(STATUS "Applying MSVC native flags: ${NATIVE_FLAGS}")
      add_compile_options(${NATIVE_FLAGS})
    else()
      message(WARNING "Unknown compiler ${CMAKE_CXX_COMPILER_ID}. POPNET_NATIVE ignored.")
    endif()

    # Autodetect L1 Cache Size

    if("${POPNET_L1_CACHE_KB}" STREQUAL "")
      if(UNIX AND EXISTS "/sys/devices/system/cpu/cpu0/cache/index0/size")
        execute_process(
          COMMAND bash -c "cat /sys/devices/system/cpu/cpu0/cache/index0/size | tr -d 'K'"
          OUTPUT_VARIABLE DETECTED_L1_CACHE_KB
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
      elseif(APPLE)
        execute_process(
          COMMAND sysctl -n hw.l1dcachesize
          OUTPUT_VARIABLE DETECTED_L1_CACHE_BYTES
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        math(EXPR DETECTED_L1_CACHE_KB "${DETECTED_L1_CACHE_BYTES} / 1024")
      elseif(WIN32)
        execute_process(
          COMMAND powershell -Command "(Get-CimInstance Win32_CacheMemory | Where-Object { $_.Level -eq 3 }).InstalledSize"
          OUTPUT_VARIABLE L1_TOTAL_KB
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(REGEX MATCH "^[0-9]+" L1_TOTAL_KB "${L1_TOTAL_KB}")
        execute_process(
          COMMAND powershell -Command "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"
          OUTPUT_VARIABLE CPU_PHYSICAL_CORES
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(REGEX MATCH "^[0-9]+" CPU_PHYSICAL_CORES "${CPU_PHYSICAL_CORES}")
        if(L1_TOTAL_KB AND CPU_PHYSICAL_CORES AND NOT CPU_PHYSICAL_CORES STREQUAL "0")
          math(EXPR DETECTED_L1_CACHE_KB "${L1_TOTAL_KB} / ${CPU_PHYSICAL_CORES}")
        endif()
      endif()
      if(DETECTED_L1_CACHE_KB)
        message(STATUS "Applying native l1 cache size: ${DETECTED_L1_CACHE_KB} KB")
        set(POPNET_L1_CACHE_KB ${DETECTED_L1_CACHE_KB})
      endif()
    endif()
  endif()
endif()