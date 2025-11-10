#include "reduce.h"

#include "simd.h"
#include "tensor.h"

template <auto init_state_op, auto merge_op, auto horizontal_op, popnet_type dtype>
void reduce_kernel(void* POPNET_RESTRICT src, void* POPNET_RESTRICT dst, size_t n) noexcept
{
  using traits = simd_traits<dtype>;
  using scalar_t = typename traits::scalar_t;

  *static_cast<scalar_t*>(dst) = simd_reduce<init_state_op, merge_op, horizontal_op, POPNET_TYPE_F32>(n, src);
}

template <popnet_op op> POPNET_INLINE static void popnet_compute_reduce(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];
  const size_t n = popnet_nelems(src->shape);

  if (src->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    if constexpr (op == POPNET_OP_REDUCE_SUM)
      reduce_kernel<vset_zero_f32, vadd_f32, vreduce_sum_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_REDUCE_MEAN)
      reduce_kernel<vset_zero_f32, vadd_f32, vreduce_mean_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_REDUCE_PRODUCT)
      reduce_kernel<vset_one_f32, vmul_f32, vreduce_product_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_REDUCE_MIN)
      reduce_kernel<vset_pos_inf_f32, vmin_f32, vreduce_min_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_REDUCE_MAX)
      reduce_kernel<vset_neg_inf_f32, vmax_f32, vreduce_max_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else
      static_assert([]() { return false; }, "Not implemented");
  }
  else
  {
    POPNET_UNREACHABLE(); // TODO
  }
}

void popnet_compute_reduce_sum(popnet_tensor* dst) { popnet_compute_reduce<POPNET_OP_REDUCE_SUM>(dst); };
void popnet_compute_reduce_mean(popnet_tensor* dst) { popnet_compute_reduce<POPNET_OP_REDUCE_MEAN>(dst); };
void popnet_compute_reduce_product(popnet_tensor* dst) { popnet_compute_reduce<POPNET_OP_REDUCE_PRODUCT>(dst); };
void popnet_compute_reduce_min(popnet_tensor* dst) { popnet_compute_reduce<POPNET_OP_REDUCE_MIN>(dst); };
void popnet_compute_reduce_max(popnet_tensor* dst) { popnet_compute_reduce<POPNET_OP_REDUCE_MAX>(dst); };
