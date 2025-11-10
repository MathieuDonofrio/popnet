#include "linalg.h"

#include "simd.h"
#include "tensor.h"

consteval size_t sqrti(size_t x)
{
  size_t lo = 0;
  size_t hi = static_cast<size_t>(x);
  while (lo * lo < hi)
  {
    size_t mid = (lo + hi) / 2;
    if (mid * mid <= x) lo = mid + 1;
    else
      hi = mid;
  }
  return lo - 1;
}

consteval size_t compute_transpose_tile_size(size_t element_size)
{
  return sqrti((POPNET_L1_CACHE_KB * 1024) / (element_size * 2)); // Half L1 for src and dst tiles
}

template <popnet_type dtype>
void transpose_kernel(void* POPNET_RESTRICT src, void* POPNET_RESTRICT dst, size_t outer_size, size_t plane_size,
                      size_t n, size_t m)
{
  using traits = simd_traits<dtype>;
  using scalar_t = traits::scalar_t;

  static constexpr size_t tile = compute_transpose_tile_size(sizeof(scalar_t));

  const auto src_data = static_cast<const scalar_t*>(src);
  const auto dst_data = static_cast<scalar_t*>(dst);

  for (size_t outer = 0; outer < outer_size; ++outer)
  {
    const auto src_plane_data = src_data + outer * plane_size;
    const auto dst_plane_data = dst_data + outer * plane_size;

    for (size_t i0 = 0; i0 < n; i0 += tile)
    {
      for (size_t j0 = 0; j0 < m; j0 += tile)
      {
        const auto i_max = std::min(i0 + tile, n);
        const auto j_max = std::min(j0 + tile, m);

        for (size_t i = i0; i < i_max; ++i)
        {
          for (size_t j = j0; j < j_max; ++j)
          {
            dst_plane_data[j * n + i] = src_plane_data[i * m + j];
          }
        }
      }
    }
  }
}

void popnet_compute_transpose(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];

  const size_t outer_size = src->shape.dim[0] * src->shape.dim[1];
  const size_t n = src->shape.dim[2];
  const size_t m = src->shape.dim[3];
  const size_t plane_size = m * n;

  if (src->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    transpose_kernel<POPNET_TYPE_F32>(src->data, dst->data, outer_size, plane_size, n, m);
  }
  else
  {
    POPNET_UNREACHABLE();
  }
}
