function(popnet_detect_l1_cache size_out)
  if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    file(GLOB cache_entries "/sys/devices/system/cpu/cpu0/cache/index*")
    foreach(entry IN LISTS cache_entries)
      if(NOT EXISTS "${entry}/level" OR NOT EXISTS "${entry}/type" OR NOT EXISTS "${entry}/size")
        continue()
      endif()

      file(STRINGS "${entry}/level" level LIMIT_COUNT 1)
      file(STRINGS "${entry}/type" type LIMIT_COUNT 1)
      if(level STREQUAL "1" AND type MATCHES "^(Data|Unified)$")
        file(STRINGS "${entry}/size" cache_size LIMIT_COUNT 1)
        if(cache_size MATCHES "^([1-9][0-9]*)K$")
          set(size "${CMAKE_MATCH_1}")
          break()
        endif()
      endif()
    endforeach()
  elseif(CMAKE_HOST_APPLE)
    execute_process(
      COMMAND sysctl -n hw.l1dcachesize
      OUTPUT_VARIABLE size_bytes
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(size_bytes MATCHES "^[1-9][0-9]*$")
      math(EXPR size "${size_bytes} / 1024")
    endif()
  elseif(CMAKE_HOST_WIN32)
    set(probe "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/popnet-l1-cache.cpp")
    file(WRITE "${probe}" [=[
#include <cstdio>
#include <vector>
#include <windows.h>

int main()
{
  DWORD bytes = 0;
  if (GetLogicalProcessorInformation(nullptr, &bytes) || GetLastError() != ERROR_INSUFFICIENT_BUFFER) return 1;

  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> entries(bytes / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  if (!GetLogicalProcessorInformation(entries.data(), &bytes)) return 1;

  DWORD cache_bytes = 0;
  for (const auto& entry : entries)
  {
    if (entry.Relationship != RelationCache) continue;

    const auto& cache = entry.Cache;
    if (cache.Level == 1 && (cache.Type == CacheData || cache.Type == CacheUnified) && cache.Size != 0 &&
        (cache_bytes == 0 || cache.Size < cache_bytes))
      cache_bytes = cache.Size;
  }

  if (cache_bytes < 1024) return 1;
  std::printf("%lu", cache_bytes / 1024);
}
]=])

    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
    try_run(popnet_cache_run popnet_cache_compiled "${CMAKE_CURRENT_BINARY_DIR}" "${probe}"
      RUN_OUTPUT_VARIABLE cache_size
    )
    if(popnet_cache_compiled AND popnet_cache_run STREQUAL "0")
      set(size "${cache_size}")
    endif()
    unset(popnet_cache_run CACHE)
    unset(popnet_cache_compiled CACHE)
  endif()

  if(size MATCHES "^[1-9][0-9]*$")
    set(${size_out} "${size}" PARENT_SCOPE)
  endif()
endfunction()

function(popnet_configure_cpu_cache target)
  set(size "${POPNET_L1_CACHE_KB}")
  string(TOUPPER "${POPNET_CPU_TARGET}" target_mode)
  if(size STREQUAL "" AND target_mode STREQUAL "NATIVE")
    popnet_detect_l1_cache(size)
    if(size)
      message(STATUS "Popnet native L1 data cache: ${size} KiB")
    endif()
  endif()

  if(size STREQUAL "")
    set(size 32)
  elseif(NOT size MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "POPNET_L1_CACHE_KB must be a positive integer in KiB.")
  endif()

  target_compile_definitions(${target} PRIVATE POPNET_L1_CACHE_KB=${size})
endfunction()
