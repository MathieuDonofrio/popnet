#include "allocator.h"

#include <new>
#include <stdlib.h>

void* popnet_alloc(popnet_type dtype, popnet_shape shape)
{
  size_t nelems = popnet_nelems(shape);
  size_t size = 0;

  switch (dtype)
  {
  case POPNET_TYPE_F32: size = sizeof(float) * nelems; break;
  case POPNET_TYPE_I32: size = sizeof(int32_t) * nelems; break;
  default: POPNET_UNREACHABLE(); break;
  }

  POPNET_EXPECT(size > 0);

  // Allocate memory aligned to the hardware's cache line size for optimal SIMD access and to minimize false sharing
  // between threads.
  constexpr size_t alignment = std::hardware_destructive_interference_size;

#ifdef _MSC_VER
  void* ptr = _aligned_malloc(size, alignment);
#else
  const size_t padded_size = ((size + alignment - 1) / alignment) * alignment;

  void* ptr = aligned_alloc(alignment, padded_size);
#endif

  POPNET_EXPECT(ptr != nullptr);

  return ptr;
}

void popnet_free(void* ptr)
{
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}
