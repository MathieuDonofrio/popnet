#include "elemwise.h"

#include "simd.h"
#include "tensor.h"

template <auto op, popnet_type dtype>
void unary_kernel(void* POPNET_RESTRICT src, void* POPNET_RESTRICT dst, size_t n) noexcept
{
  simd_map<op, dtype>(dst, n, src);
}

template <auto op, popnet_type dtype>
void binary_kernel(void* POPNET_RESTRICT src0, void* POPNET_RESTRICT src1, void* POPNET_RESTRICT dst, size_t n) noexcept
{
  // TODO handle broadcasting

  simd_map<op, dtype>(dst, n, src0, src1);
}

template <popnet_op op> POPNET_INLINE static void popnet_compute_unary(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];
  const size_t n = popnet_nelems(dst->shape);

  if (src->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    if constexpr (op == POPNET_OP_ABS) //
      unary_kernel<vabs_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_NEG)
      unary_kernel<vneg_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_SQR)
      unary_kernel<vsqr_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_SQRT)
      unary_kernel<vsqrt_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_EXP)
      unary_kernel<vexp_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_LOG)
      unary_kernel<vlog_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_SIN)
      unary_kernel<vsin_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_COS)
      unary_kernel<vcos_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_TANH)
      unary_kernel<vtanh_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_RELU)
      unary_kernel<vrelu_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else if constexpr (op == POPNET_OP_SIGMOID)
      unary_kernel<vsigmoid_f32, POPNET_TYPE_F32>(src->data, dst->data, n);
    else
      static_assert([]() { return false; }, "Not implemented");
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
    if constexpr (op == POPNET_OP_ADD) //
      binary_kernel<vadd_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, n);
    else if constexpr (op == POPNET_OP_SUB)
      binary_kernel<vsub_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, n);
    else if constexpr (op == POPNET_OP_MUL)
      binary_kernel<vmul_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, n);
    else if constexpr (op == POPNET_OP_DIV)
      binary_kernel<vdiv_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, n);
    else
      static_assert([]() { return false; }, "Not implemented");
  }
  else
  {
    POPNET_UNREACHABLE(); // TODO
  }
}

void popnet_compute_add(popnet_tensor* dst) { popnet_compute_binary<POPNET_OP_ADD>(dst); }
void popnet_compute_sub(popnet_tensor* dst) { popnet_compute_binary<POPNET_OP_SUB>(dst); }
void popnet_compute_mul(popnet_tensor* dst) { popnet_compute_binary<POPNET_OP_MUL>(dst); }
void popnet_compute_div(popnet_tensor* dst) { popnet_compute_binary<POPNET_OP_DIV>(dst); }
void popnet_compute_abs(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_ABS>(dst); }
void popnet_compute_neg(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_NEG>(dst); }
void popnet_compute_sqr(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_SQR>(dst); }
void popnet_compute_sqrt(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_SQRT>(dst); }
void popnet_compute_exp(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_EXP>(dst); }
void popnet_compute_log(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_LOG>(dst); }
void popnet_compute_sin(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_SIN>(dst); }
void popnet_compute_cos(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_COS>(dst); }
void popnet_compute_tanh(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_TANH>(dst); }
void popnet_compute_relu(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_RELU>(dst); }
void popnet_compute_sigmoid(popnet_tensor* dst) { popnet_compute_unary<POPNET_OP_SIGMOID>(dst); }
