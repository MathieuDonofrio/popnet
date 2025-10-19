#include "allocator.h"

#include <cstdlib>

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

  return malloc(size);
}

void popnet_free(void* ptr) { free(ptr); }
