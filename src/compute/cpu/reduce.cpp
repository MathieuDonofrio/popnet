#include "reduce.h"

#include "dtype_traits.h"
#include "memory.h"
#include "simd_algorithms.h"
#include "tensor.h"

template <auto init_state_op, auto merge_op, auto horizontal_op, popnet_dtype dtype>
POPNET_NO_INLINE void reduce_kernel_contiguous(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b,
                                               size_t outer_size, size_t reduce_size) noexcept
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = typename dtype_traits::scalar_t;

  const auto* src = static_cast<const scalar_t*>(a);
  auto* dst = static_cast<scalar_t*>(b);

  for (size_t outer = 0; outer < outer_size; ++outer)
  {
    dst[outer] =
        simd_reduce<init_state_op, merge_op, horizontal_op, popnet_dtype_f32>(reduce_size, src + outer * reduce_size);
  }
}

template <auto init_state_op, auto merge_op, popnet_dtype dtype>
POPNET_NO_INLINE void reduce_kernel_rows(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t outer_size,
                                         size_t reduce_size, size_t inner_size) noexcept
{
  using traits = dtype_traits<dtype>;
  using scalar_t = typename traits::scalar_t;
  const auto* src = static_cast<const scalar_t*>(a);
  auto* dst = static_cast<scalar_t*>(b);

  for (size_t outer = 0; outer < outer_size; ++outer)
    simd_reduce_rows<init_state_op, merge_op, dtype>(dst + outer * inner_size, reduce_size, inner_size,
                                                     src + outer * reduce_size * inner_size);
}

struct reduction_group
{
  size_t outer_size;
  size_t reduce_size;
  size_t inner_size;
};

POPNET_INLINE static reduction_group find_reduction_group(const popnet_shape src, const popnet_shape dst)
{
  POPNET_EXPECT(popnet_valid_shape(src) && popnet_valid_shape(dst));
  for (size_t i = 0; i < POPNET_MAX_DIMS; ++i)
    POPNET_EXPECT(dst.dim[i] == src.dim[i] || dst.dim[i] == 1);

  size_t begin = 0;
  while (begin < POPNET_MAX_DIMS && src.dim[begin] == dst.dim[begin])
    ++begin;
  POPNET_EXPECT(begin < POPNET_MAX_DIMS);

  size_t end = POPNET_MAX_DIMS;
  while (src.dim[end - 1] == dst.dim[end - 1])
    --end;

  reduction_group group{1, 1, popnet_inner_numel(src, end)};
  for (size_t i = 0; i < begin; ++i)
    group.outer_size *= src.dim[i];
  for (size_t i = begin; i < end; ++i)
  {
    if (dst.dim[i] != 1) POPNET_UNSUPPORTED("Separated reduction groups must be composed as separate operations");
    group.reduce_size *= src.dim[i];
  }
  return group;
}

template <auto init_state_op, auto merge_op, auto horizontal_op, popnet_dtype dtype>
POPNET_INLINE static void execute_reduction(const void* src, void* dst, const reduction_group& group)
{
  if (group.inner_size == 1)
    reduce_kernel_contiguous<init_state_op, merge_op, horizontal_op, dtype>(src, dst, group.outer_size,
                                                                            group.reduce_size);
  else
    reduce_kernel_rows<init_state_op, merge_op, dtype>(src, dst, group.outer_size, group.reduce_size, group.inner_size);
}
template <popnet_op op> POPNET_INLINE static void popnet_compute_reduce(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];

  if (src->dtype == popnet_dtype_f32 && dst->dtype == popnet_dtype_f32)
  {
    if (src->shape == dst->shape)
    {
      const size_t n = popnet_numel(src->shape);

      // Singleton-axis reductions preserve values bit-for-bit, including signed zero and NaNs.
      popnet_memcpy(dst->data, src->data, n * sizeof(float), memcpy_kind::device_to_device);
      return;
    }
    const auto group = find_reduction_group(src->shape, dst->shape);

    if constexpr (op == popnet_op_reduce_sum || op == popnet_op_reduce_mean)
      execute_reduction<simd::set_zero_f32, simd::add_f32, simd::reduce_sum_f32, popnet_dtype_f32>(src->data, dst->data,
                                                                                                   group);
    else if constexpr (op == popnet_op_reduce_product)
      execute_reduction<simd::set_one_f32, simd::mul_f32, simd::reduce_product_f32, popnet_dtype_f32>(src->data,
                                                                                                      dst->data, group);
    else if constexpr (op == popnet_op_reduce_min)
      execute_reduction<simd::set_pos_inf_f32, simd::min_f32, simd::reduce_min_f32, popnet_dtype_f32>(src->data,
                                                                                                      dst->data, group);
    else if constexpr (op == popnet_op_reduce_max)
      execute_reduction<simd::set_neg_inf_f32, simd::max_f32, simd::reduce_max_f32, popnet_dtype_f32>(src->data,
                                                                                                      dst->data, group);
    else
      POPNET_UNSUPPORTED_STATIC("Unsupported reduction operation");

    if constexpr (op == popnet_op_reduce_mean)
    {
      const size_t output_size = popnet_numel(dst->shape);
      const size_t reduction_count = popnet_numel(src->shape) / output_size;
      const float divisor = static_cast<float>(reduction_count);
      simd_map_inplace<simd::div_f32, popnet_dtype_f32>(
          dst->data, output_size, static_cast<const broadcast*>(static_cast<const void*>(&divisor)));
    }
  }
  else
  {
    POPNET_UNSUPPORTED("Reduction requires f32 input and output");
  }
}

void popnet_compute_reduce_sum(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_sum>(dst); };
void popnet_compute_reduce_mean(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_mean>(dst); };
void popnet_compute_reduce_product(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_product>(dst); };
void popnet_compute_reduce_min(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_min>(dst); };
void popnet_compute_reduce_max(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_max>(dst); };
