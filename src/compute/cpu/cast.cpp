#include "cast.h"

#include <algorithm>

#include "dtype.h"
#include "tensor.h"
#include "ternary.h"

namespace
{
template <typename T> void pack_block(const T* src, uint8_t* dst, size_t count)
{
  const size_t plane_bytes = ternary::plane_bytes(count);
  auto* negative = dst + plane_bytes;
  std::fill_n(dst, 2 * plane_bytes, uint8_t{0});

  for (size_t i = 0; i < count; ++i)
  {
    const T value = src[i];
    POPNET_EXPECT(value == -1 || value == 0 || value == 1);
    const auto mask = static_cast<uint8_t>(1u << (i % 8));
    if (value != 0) dst[i / 8] |= mask;
    if (value < 0) negative[i / 8] |= mask;
  }
}

template <typename T> void unpack_block(const uint8_t* src, T* dst, size_t count)
{
  const auto* negative = src + ternary::plane_bytes(count);
  for (size_t i = 0; i < count; ++i)
  {
    const unsigned mask = 1u << (i % 8);
    const bool nonzero = (src[i / 8] & mask) != 0;
    const bool neg = (negative[i / 8] & mask) != 0;
    dst[i] = static_cast<T>(neg ? -1 : nonzero);
  }
}
template <typename Src, typename Dst>
void cast_packed(const popnet_tensor* input, popnet_tensor* output, void (*convert)(const Src*, Dst*, size_t))
{
  auto* src = static_cast<const Src*>(input->data);
  auto* dst = static_cast<Dst*>(output->data);
  const size_t width = input->shape.dim[3];
  const size_t rows = popnet_numel(input->shape) / width;

  for (size_t row = 0; row < rows; ++row)
  {
    for (size_t col = 0; col < width;)
    {
      const size_t count = std::min(width - col, ternary::block_elements);
      const size_t bytes = ternary::row_bytes(count);
      convert(src, dst, count);
      src += input->dtype == popnet_dtype_ternary ? bytes : count;
      dst += output->dtype == popnet_dtype_ternary ? bytes : count;
      col += count;
    }
  }
}

template <typename Src, typename Dst>
void cast_unpacked(const Src* POPNET_RESTRICT src, Dst* POPNET_RESTRICT dst, size_t count)
{
  for (size_t i = 0; i < count; ++i)
  {
    dst[i] = static_cast<Dst>(src[i]);
  }
}
} // namespace

void popnet_compute_cast(popnet_tensor* dst)
{
  const auto* src = dst->src[0];
  switch (dtype_pair_key(src->dtype, dst->dtype))
  {
  case dtype_pair_key<popnet_dtype_f32, popnet_dtype_i8>():
  {
    cast_unpacked(static_cast<const float*>(src->data), static_cast<int8_t*>(dst->data), popnet_numel(src->shape));
    break;
  }
  case dtype_pair_key<popnet_dtype_i8, popnet_dtype_f32>():
  {
    cast_unpacked(static_cast<const int8_t*>(src->data), static_cast<float*>(dst->data), popnet_numel(src->shape));
    break;
  }
  case dtype_pair_key<popnet_dtype_f32, popnet_dtype_ternary>():
  {
    cast_packed(src, dst, pack_block<float>);
    break;
  }
  case dtype_pair_key<popnet_dtype_i8, popnet_dtype_ternary>():
  {
    cast_packed(src, dst, pack_block<int8_t>);
    break;
  }
  case dtype_pair_key<popnet_dtype_ternary, popnet_dtype_f32>():
  {
    cast_packed(src, dst, unpack_block<float>);
    break;
  }
  case dtype_pair_key<popnet_dtype_ternary, popnet_dtype_i8>():
  {
    cast_packed(src, dst, unpack_block<int8_t>);
    break;
  }
  default: POPNET_UNSUPPORTED("Unsupported cast dtype pair"); break;
  }
}
