#include "elemwise.h"

#include "simd.h"
#include "tensor.h"

///
/// Templated Kernels
///

template <auto UnaryOpFunc> POPNET_INLINE static void popnet_compute_unary(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];
  const size_t n = popnet_nelems(dst->shape);

  if (src->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    vtransform<UnaryOpFunc>(static_cast<vf32*>(dst->data), n, static_cast<const vf32*>(src->data));
  }
  else
  {
    POPNET_UNREACHABLE(); // TODO
  }
}

template <auto BinaryOpFunc> POPNET_INLINE static void popnet_compute_binary(popnet_tensor* dst)
{
  const popnet_tensor* src0 = dst->src[0];
  const popnet_tensor* src1 = dst->src[1];

  const size_t n = popnet_nelems(dst->shape);

  if (src0->dtype == POPNET_TYPE_F32 && src1->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    vtransform<BinaryOpFunc>(static_cast<vf32*>(dst->data), n, static_cast<const vf32*>(src0->data),
                             static_cast<const vf32*>(src1->data));
  }
  else
  {
    POPNET_UNREACHABLE(); // TODO
  }
}

///
/// Kernels
///

void popnet_compute_add(popnet_tensor* dst) { popnet_compute_binary<vadd>(dst); }
void popnet_compute_sub(popnet_tensor* dst) { popnet_compute_binary<vsub>(dst); }
void popnet_compute_mul(popnet_tensor* dst) { popnet_compute_binary<vmul>(dst); }
void popnet_compute_div(popnet_tensor* dst) { popnet_compute_binary<vdiv>(dst); }
void popnet_compute_abs(popnet_tensor* dst) { popnet_compute_unary<vabs>(dst); }
void popnet_compute_neg(popnet_tensor* dst) { popnet_compute_unary<vneg>(dst); }
void popnet_compute_sqr(popnet_tensor* dst) { popnet_compute_unary<vsqr>(dst); }
void popnet_compute_sqrt(popnet_tensor* dst) { popnet_compute_unary<vsqrt>(dst); }
void popnet_compute_exp(popnet_tensor* dst) { popnet_compute_unary<vexp>(dst); }
void popnet_compute_log(popnet_tensor* dst) { popnet_compute_unary<vlog>(dst); }
void popnet_compute_sin(popnet_tensor* dst) { popnet_compute_unary<vsin>(dst); }
void popnet_compute_cos(popnet_tensor* dst) { popnet_compute_unary<vcos>(dst); }
void popnet_compute_tanh(popnet_tensor* dst) { popnet_compute_unary<vtanh>(dst); }
void popnet_compute_relu(popnet_tensor* dst) { popnet_compute_unary<vrelu>(dst); }
void popnet_compute_sigmoid(popnet_tensor* dst) { popnet_compute_unary<vsigmoid>(dst); }
