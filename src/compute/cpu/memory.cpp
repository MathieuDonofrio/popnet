#include "memory.h"

#include <new>
#include <stdlib.h>
#include <string.h>

void* popnet_alloc(popnet_dtype dtype, popnet_shape shape) noexcept
{
  const size_t nelems = popnet_numel(shape);

  if (nelems == 0) return nullptr;

  size_t size = 0;

  switch (dtype)
  {
  case popnet_dtype_f32:
  {
    if (nelems > SIZE_MAX / sizeof(float)) return nullptr;
    size = sizeof(float) * nelems;
    break;
  }
  default: POPNET_UNSUPPORTED("CPU allocation requires f32 dtype"); break;
  }

  POPNET_EXPECT(size > 0);

  // Separate allocations on cache-line-sized boundaries; use a fallback when the library exposes no hint.
#ifndef __cpp_lib_hardware_interference_size
  constexpr size_t alignment = 64;
#else
  constexpr size_t alignment = std::hardware_destructive_interference_size;
#endif

#ifdef _MSC_VER
  void* ptr = _aligned_malloc(size, alignment);
#else
  // aligned_alloc requires the allocation size to be a multiple of its alignment.
  const size_t remainder = size % alignment;
  const size_t padding = remainder ? alignment - remainder : 0;

  if (size > SIZE_MAX - padding) return nullptr;

  const size_t padded_size = size + padding;

  void* ptr = aligned_alloc(alignment, padded_size);
#endif

  return ptr;
}

void popnet_free(void* ptr) noexcept
{
  if (!ptr) return;

#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

void popnet_memcpy(void* POPNET_RESTRICT dst, const void* POPNET_RESTRICT src, size_t count,
                   [[maybe_unused]] memcpy_kind kind) noexcept
{
  memcpy(dst, src, count);
}
