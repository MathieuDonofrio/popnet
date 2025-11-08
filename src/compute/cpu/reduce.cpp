#include "reduce.h"

#include "simd.h"
#include "tensor.h"

template <auto init_state_op, auto merge_op, auto horizontal_op>
POPNET_INLINE static void popnet_compute_reduce(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];
  const size_t n = popnet_nelems(dst->shape);

  if (src->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    // TODO handle any axis
    *static_cast<float*>(dst->data) =
        simd_reduce<init_state_op, merge_op, horizontal_op, POPNET_TYPE_F32>(n, src->data);
  }
  else
  {
    POPNET_UNREACHABLE(); // TODO
  }
}

void popnet_compute_reduce_sum(popnet_tensor* dst)
{
  popnet_compute_reduce<vset_zero_f32, vadd_f32, vreduce_sum_f32>(dst);
};

void popnet_compute_reduce_mean(popnet_tensor* dst)
{
  popnet_compute_reduce<vset_zero_f32, vadd_f32, vreduce_mean_f32>(dst);
};

void popnet_compute_reduce_product(popnet_tensor* dst)
{
  popnet_compute_reduce<vset_one_f32, vmul_f32, vreduce_product_f32>(dst);
};

void popnet_compute_reduce_min(popnet_tensor* dst)
{
  popnet_compute_reduce<vset_pos_inf_f32, vmin_f32, vreduce_min_f32>(dst);
};

void popnet_compute_reduce_max(popnet_tensor* dst)
{
  popnet_compute_reduce<vset_neg_inf_f32, vmax_f32, vreduce_max_f32>(dst);
};
