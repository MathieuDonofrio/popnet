#include "reduce.h"

#include "dtype_traits.h"
#include "memory.h"
#include "simd_algorithms.h"
#include "tensor.h"

template <auto init_state_op, auto merge_op, auto horizontal_op, popnet_dtype dtype>
POPNET_NO_INLINE void reduce_kernel_scalar(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n) noexcept
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = typename dtype_traits::scalar_t;

  *static_cast<scalar_t*>(b) = simd_reduce<init_state_op, merge_op, horizontal_op, popnet_dtype_f32>(n, a);
}

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

POPNET_INLINE static size_t first_reduction_dim(popnet_shape src_shape, popnet_shape dst_shape)
{
  for (size_t i = 0; i < POPNET_MAX_DIMS; ++i)
  {
    if (src_shape.dim[i] == dst_shape.dim[i]) continue;
    if (dst_shape.dim[i] != 1) break;
    return i;
  }

  return POPNET_MAX_DIMS;
}

POPNET_INLINE static bool is_contiguous_inner_reduction(const popnet_shape dst_shape, size_t reduce_start)
{
  for (size_t i = reduce_start; i < POPNET_MAX_DIMS; ++i)
  {
    if (dst_shape.dim[i] != 1) return false;
  }

  return true;
}

template <popnet_op op> POPNET_INLINE static void popnet_compute_reduce(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];

  const size_t first_reduce_dim = first_reduction_dim(src->shape, dst->shape);

  if (src->dtype == popnet_dtype_f32 && dst->dtype == popnet_dtype_f32)
  {
    if (first_reduce_dim == POPNET_MAX_DIMS)
    {
      const size_t n = popnet_numel(src->shape);

      if (dst->shape == popnet_scalar_shape)
      {
        if constexpr (op == popnet_op_reduce_sum)
          reduce_kernel_scalar<vset_zero_f32, vadd_f32, vreduce_sum_f32, popnet_dtype_f32>(src->data, dst->data, n);
        else if constexpr (op == popnet_op_reduce_mean)
          reduce_kernel_scalar<vset_zero_f32, vadd_f32, vreduce_mean_f32, popnet_dtype_f32>(src->data, dst->data, n);
        else if constexpr (op == popnet_op_reduce_product)
          reduce_kernel_scalar<vset_one_f32, vmul_f32, vreduce_product_f32, popnet_dtype_f32>(src->data, dst->data, n);
        else if constexpr (op == popnet_op_reduce_min)
          reduce_kernel_scalar<vset_pos_inf_f32, vmin_f32, vreduce_min_f32, popnet_dtype_f32>(src->data, dst->data, n);
        else if constexpr (op == popnet_op_reduce_max)
          reduce_kernel_scalar<vset_neg_inf_f32, vmax_f32, vreduce_max_f32, popnet_dtype_f32>(src->data, dst->data, n);
        else
          POPNET_UNSUPPORTED_STATIC("Unsupported reduction operation");
      }
      else
      {
        popnet_memcpy(dst->data, src->data, n * sizeof(float), memcpy_kind::device_to_device);
      }
    }
    else if (is_contiguous_inner_reduction(dst->shape, first_reduce_dim + 1))
    {
      const size_t reduce_size = popnet_inner_numel(src->shape, first_reduce_dim);
      const size_t outer_size = popnet_numel(dst->shape);

      if constexpr (op == popnet_op_reduce_sum)
        reduce_kernel_contiguous<vset_zero_f32, vadd_f32, vreduce_sum_f32, popnet_dtype_f32>(src->data, dst->data,
                                                                                             outer_size, reduce_size);
      else if constexpr (op == popnet_op_reduce_mean)
        reduce_kernel_contiguous<vset_zero_f32, vadd_f32, vreduce_mean_f32, popnet_dtype_f32>(src->data, dst->data,
                                                                                              outer_size, reduce_size);
      else if constexpr (op == popnet_op_reduce_product)
        reduce_kernel_contiguous<vset_one_f32, vmul_f32, vreduce_product_f32, popnet_dtype_f32>(
            src->data, dst->data, outer_size, reduce_size);
      else if constexpr (op == popnet_op_reduce_min)
        reduce_kernel_contiguous<vset_pos_inf_f32, vmin_f32, vreduce_min_f32, popnet_dtype_f32>(
            src->data, dst->data, outer_size, reduce_size);
      else if constexpr (op == popnet_op_reduce_max)
        reduce_kernel_contiguous<vset_neg_inf_f32, vmax_f32, vreduce_max_f32, popnet_dtype_f32>(
            src->data, dst->data, outer_size, reduce_size);
      else
        POPNET_UNSUPPORTED_STATIC("Unsupported reduction operation");
    }
    else
    {
      POPNET_UNSUPPORTED("Reduction over non-contiguous inner dimensions is unsupported");
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
