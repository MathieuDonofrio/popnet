#include "elemwise.h"

#include "simd.h"
#include "tensor.h"

template <auto op> POPNET_INLINE static void popnet_compute_unary(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];
  const size_t n = popnet_nelems(dst->shape);

  if (src->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    simd_map<op, POPNET_TYPE_F32>(dst->data, n, src->data);
  }
  else
  {
    POPNET_UNREACHABLE(); // TODO
  }
}

template <auto op> POPNET_INLINE static void popnet_compute_binary(popnet_tensor* dst)
{
  const popnet_tensor* src0 = dst->src[0];
  const popnet_tensor* src1 = dst->src[1];

  const size_t n = popnet_nelems(dst->shape);

  if (src0->dtype == POPNET_TYPE_F32 && src1->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    // TODO handle broadcasting
    simd_map<op, POPNET_TYPE_F32>(dst->data, n, src0->data, src1->data);
  }
  else
  {
    POPNET_UNREACHABLE(); // TODO
  }
}

void popnet_compute_add(popnet_tensor* dst) { popnet_compute_binary<vadd_f32>(dst); }
void popnet_compute_sub(popnet_tensor* dst) { popnet_compute_binary<vsub_f32>(dst); }
void popnet_compute_mul(popnet_tensor* dst) { popnet_compute_binary<vmul_f32>(dst); }
void popnet_compute_div(popnet_tensor* dst) { popnet_compute_binary<vdiv_f32>(dst); }
void popnet_compute_abs(popnet_tensor* dst) { popnet_compute_unary<vabs_f32>(dst); }
void popnet_compute_neg(popnet_tensor* dst) { popnet_compute_unary<vneg_f32>(dst); }
void popnet_compute_sqr(popnet_tensor* dst) { popnet_compute_unary<vsqr_f32>(dst); }
void popnet_compute_sqrt(popnet_tensor* dst) { popnet_compute_unary<vsqrt_f32>(dst); }
void popnet_compute_exp(popnet_tensor* dst) { popnet_compute_unary<vexp_f32>(dst); }
void popnet_compute_log(popnet_tensor* dst) { popnet_compute_unary<vlog_f32>(dst); }
void popnet_compute_sin(popnet_tensor* dst) { popnet_compute_unary<vsin_f32>(dst); }
void popnet_compute_cos(popnet_tensor* dst) { popnet_compute_unary<vcos_f32>(dst); }
void popnet_compute_tanh(popnet_tensor* dst) { popnet_compute_unary<vtanh_f32>(dst); }
void popnet_compute_relu(popnet_tensor* dst) { popnet_compute_unary<vrelu_f32>(dst); }
void popnet_compute_sigmoid(popnet_tensor* dst) { popnet_compute_unary<vsigmoid_f32>(dst); }
