#include "linalg.h"

#include "dtype_traits.h"
#include "simd_algorithms.h"
#include "tensor.h"

consteval size_t sqrti(size_t x)
{
  size_t lo = 0;
  size_t hi = x;
  while (lo < hi)
  {
    const size_t mid = lo + (hi - lo) / 2 + 1;
    if (mid <= x / mid) lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

static_assert(sqrti(0) == 0 && sqrti(1) == 1 && sqrti(2) == 1);
static_assert(sqrti(15) == 3 && sqrti(16) == 4 && sqrti(17) == 4);
static_assert(sqrti(2730) == 52 && sqrti(5461) == 73 && sqrti(8192) == 90);

constexpr size_t l1_cache_size() noexcept
{
#ifdef POPNET_L1_CACHE_KB
  return POPNET_L1_CACHE_KB * 1024;
#else
  // A 32 KiB budget favors smaller tiles when cache size is unknown, trading reuse
  // on larger caches for a smaller working set. This is a heuristic, not a cache-size guarantee.
  return 32 * 1024;
#endif
}

consteval size_t compute_matmul_tile_size(size_t element_size) { return sqrti(l1_cache_size() / (element_size * 3)); }

consteval size_t compute_transpose_tile_size(size_t element_size)
{
  return sqrti(l1_cache_size() / (element_size * 2));
}

template <auto init_state_op, auto merge_op, auto horizontal_op, popnet_dtype dtype>
POPNET_NO_INLINE void matmul_kernel(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b,
                                    void* POPNET_RESTRICT c, size_t outer_size, size_t n, size_t m, size_t k)
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = dtype_traits::scalar_t;

  static constexpr size_t tile = compute_matmul_tile_size(sizeof(scalar_t));

  const auto aa = static_cast<const scalar_t*>(a);
  const auto bb = static_cast<const scalar_t*>(b);
  const auto cc = static_cast<scalar_t*>(c);

  const size_t a_plane_size = m * k;
  const size_t c_plane_size = m * n;

  for (size_t outer = 0; outer < outer_size; ++outer)
  {
    const auto a_plane = aa + outer * a_plane_size;
    auto c_plane = cc + outer * c_plane_size;

    for (size_t i0 = 0; i0 < m; i0 += tile)
    {
      for (size_t j0 = 0; j0 < n; j0 += tile)
      {
        const size_t i_max = std::min(i0 + tile, m);
        const size_t j_max = std::min(j0 + tile, n);

        for (size_t i = i0; i < i_max; ++i)
          for (size_t j = j0; j < j_max; ++j)
            c_plane[i * n + j] = 0;

        for (size_t k0 = 0; k0 < k; k0 += tile)
        {
          const size_t k_max = std::min(k0 + tile, k);

          for (size_t i = i0; i < i_max; ++i)
          {
            const auto* a_block = a_plane + i * k + k0;

            for (size_t j = j0; j < j_max; ++j)
            {
              const auto* b_block = bb + j * k + k0;
              auto* c_val = c_plane + i * n + j;

              *c_val += simd_reduce<init_state_op, merge_op, horizontal_op, dtype>(
                  k_max - k0, static_cast<const void*>(a_block), static_cast<const void*>(b_block));
            }
          }
        }
      }
    }
  }
}

template <popnet_dtype dtype>
POPNET_NO_INLINE void transpose_kernel(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t outer_size,
                                       size_t n, size_t m) noexcept
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = dtype_traits::scalar_t;

  static constexpr size_t tile = compute_transpose_tile_size(sizeof(scalar_t));

  const auto aa = static_cast<const scalar_t*>(a);
  const auto bb = static_cast<scalar_t*>(b);

  const size_t plane_size = m * n;

  for (size_t outer = 0; outer < outer_size; ++outer)
  {
    const auto a_plane = aa + outer * plane_size;
    const auto b_plane = bb + outer * plane_size;

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
            b_plane[j * n + i] = a_plane[i * m + j];
          }
        }
      }
    }
  }
}

void popnet_compute_matmul(popnet_tensor* dst)
{
  const popnet_tensor* src0 = dst->src[0];
  const popnet_tensor* src1 = dst->src[1];

  const size_t outer_size = dst->shape.dim[0] * dst->shape.dim[1];
  const size_t n = dst->shape.dim[3];
  const size_t m = dst->shape.dim[2];
  const size_t k = src0->shape.dim[3];

  // Expected shapes:
  //  src0 = (a, b, m, k)
  //  src1 = (1, 1, n, k)
  //  dst  = (a, b, m, n)
  POPNET_EXPECT(src0->shape.dim[0] == dst->shape.dim[0]);
  POPNET_EXPECT(src0->shape.dim[1] == dst->shape.dim[1]);
  POPNET_EXPECT(src1->shape.dim[0] == 1 && src1->shape.dim[1] == 1);
  POPNET_EXPECT(src0->shape.dim[2] == m);
  POPNET_EXPECT(src0->shape.dim[3] == k);
  POPNET_EXPECT(src1->shape.dim[2] == n);
  POPNET_EXPECT(src1->shape.dim[3] == k);
  POPNET_EXPECT(dst->shape.dim[2] == m);
  POPNET_EXPECT(dst->shape.dim[3] == n);

  if (src0->dtype == popnet_dtype_f32 && src1->dtype == popnet_dtype_f32 && dst->dtype == popnet_dtype_f32)
  {
    matmul_kernel<simd::set_zero_f32, simd::fma_f32, simd::reduce_sum_f32, popnet_dtype_f32>(
        src0->data, src1->data, dst->data, outer_size, n, m, k);
  }
  else
  {
    POPNET_UNSUPPORTED("Matrix multiplication requires f32 inputs and output");
  }
}

void popnet_compute_transpose(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];

  const size_t outer_size = src->shape.dim[0] * src->shape.dim[1];
  const size_t n = src->shape.dim[2];
  const size_t m = src->shape.dim[3];

  if (src->dtype == popnet_dtype_f32 && dst->dtype == popnet_dtype_f32)
  {
    transpose_kernel<popnet_dtype_f32>(src->data, dst->data, outer_size, n, m);
  }
  else
  {
    POPNET_UNSUPPORTED("Transpose requires f32 input and output");
  }
}
