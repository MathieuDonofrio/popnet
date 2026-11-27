#pragma once

#include "platform.h"
#include "popnet.h"

namespace ternary
{
constexpr size_t full_plane_bytes()
{
#if defined(POPNET_AVX512)
  return 64;
#elif defined(POPNET_AVX)
  return 32;
#elif defined(POPNET_SSE2) || defined(POPNET_NEON)
  return 16;
#else
  return sizeof(uint64_t);
#endif
}

constexpr size_t block_elements = 8 * full_plane_bytes();

constexpr size_t plane_bytes(size_t elements) { return elements / 8 + (elements % 8 != 0); }

constexpr size_t row_bytes(size_t row_elements) { return 2 * plane_bytes(row_elements); }

constexpr size_t storage_bytes(popnet_shape shape)
{
  // Each row occupies at most one byte per element, so a representable numel bounds this product.
  return shape.dim[0] * shape.dim[1] * shape.dim[2] * row_bytes(shape.dim[3]);
}

} // namespace ternary
