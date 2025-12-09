#include "reduce.h"

#include "tensor.h"
#include "traits.h"

template <auto init_state_op, auto merge_op, auto horizontal_op, popnet_type dtype>
POPNET_NO_INLINE void reduce_kernel_scalar(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n) noexcept
{
  using traits = traits<dtype>;
  using scalar_t = typename traits::scalar_t;

  *static_cast<scalar_t*>(b) = simd_reduce<init_state_op, merge_op, horizontal_op, popnet_type_f32>(n, a);
}

template <popnet_op op> POPNET_INLINE static void popnet_compute_reduce(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];
  const size_t n = popnet_numel(src->shape);

  if (src->dtype == popnet_type_f32 && dst->dtype == popnet_type_f32)
  {
    if (dst->shape == popnet_scalar_shape)
    {
      if constexpr (op == popnet_op_reduce_sum)
        reduce_kernel_scalar<vset_zero_f32, vadd_f32, vreduce_sum_f32, popnet_type_f32>(src->data, dst->data, n);
      else if constexpr (op == popnet_op_reduce_mean)
        reduce_kernel_scalar<vset_zero_f32, vadd_f32, vreduce_mean_f32, popnet_type_f32>(src->data, dst->data, n);
      else if constexpr (op == popnet_op_reduce_product)
        reduce_kernel_scalar<vset_one_f32, vmul_f32, vreduce_product_f32, popnet_type_f32>(src->data, dst->data, n);
      else if constexpr (op == popnet_op_reduce_min)
        reduce_kernel_scalar<vset_pos_inf_f32, vmin_f32, vreduce_min_f32, popnet_type_f32>(src->data, dst->data, n);
      else if constexpr (op == popnet_op_reduce_max)
        reduce_kernel_scalar<vset_neg_inf_f32, vmax_f32, vreduce_max_f32, popnet_type_f32>(src->data, dst->data, n);
      else
        static_assert([]() { return false; }, "Not implemented");
    }
    else
    {
      POPNET_UNREACHABLE(); // TODO
    }
  }
  else
  {
    POPNET_UNREACHABLE(); // TODO
  }
}

void popnet_compute_reduce_sum(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_sum>(dst); };
void popnet_compute_reduce_mean(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_mean>(dst); };
void popnet_compute_reduce_product(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_product>(dst); };
void popnet_compute_reduce_min(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_min>(dst); };
void popnet_compute_reduce_max(popnet_tensor* dst) { popnet_compute_reduce<popnet_op_reduce_max>(dst); };
