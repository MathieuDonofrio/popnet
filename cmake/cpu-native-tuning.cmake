if(POPNET_NATIVE)
  if(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling: ignoring POPNET_CPU_NATIVE=ON.")
    set(POPNET_CPU_NATIVE OFF CACHE BOOL "" FORCE)
  else()
    # Autodetect CPU flags

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
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

      # MSVC does not have a 'native' arch, so we need to detect the best available arch
      set(NATIVE_FLAGS "")

      # x86
      check_cxx_compiler_flag("/arch:AVX512" HAS_ARCH_AVX512)
      if(HAS_ARCH_AVX512 AND NATIVE_FLAGS STREQUAL "")
        set(NATIVE_FLAGS "/arch:AVX512")
      endif()
      check_cxx_compiler_flag("/arch:AVX2" HAS_ARCH_AVX2)
      if(HAS_ARCH_AVX2 AND NATIVE_FLAGS STREQUAL "")
        set(NATIVE_FLAGS "/arch:AVX2")
      endif()
      check_cxx_compiler_flag("/arch:AVX" HAS_ARCH_AVX)
      if(HAS_ARCH_AVX AND NATIVE_FLAGS STREQUAL "")
        set(NATIVE_FLAGS "/arch:AVX")
      endif()
      # MSVC defaults to SSE2

      if(NATIVE_FLAGS STREQUAL "")
        message(WARNING "No /arch flag detected")
      endif()

      message(STATUS "Applying MSVC native flags: ${NATIVE_FLAGS}")
      add_compile_options(${NATIVE_FLAGS})
    else()
      message(WARNING "Unknown compiler (${CMAKE_CXX_COMPILER_ID}): POPNET_NATIVE ignored.")
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