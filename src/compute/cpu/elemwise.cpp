#include "elemwise.h"

#include "tensor.h"
#include "traits.h"

template <auto op, popnet_type dtype>
POPNET_NO_INLINE void unary_kernel(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n) noexcept
{
  POPNET_EXPECT(n > 0);
  simd_map<op, dtype>(b, n, a);
}

template <auto op, popnet_type dtype>
POPNET_NO_INLINE void binary_kernel_matching(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b,
                                             void* POPNET_RESTRICT c, size_t n) noexcept
{
  POPNET_EXPECT(n > 0);
  simd_map<op, dtype>(c, n, a, b);
}

template <auto op, popnet_type dtype>
POPNET_NO_INLINE void binary_kernel_lscalar(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b,
                                            void* POPNET_RESTRICT c, size_t n) noexcept
{
  POPNET_EXPECT(n > 0);
  simd_map<op, dtype>(c, n, static_cast<const broadcast*>(a), b);
}

template <auto op, popnet_type dtype>
POPNET_NO_INLINE void binary_kernel_rscalar(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b,
                                            void* POPNET_RESTRICT c, size_t n) noexcept
{
  POPNET_EXPECT(n > 0);
  simd_map<op, dtype>(c, n, a, static_cast<const broadcast*>(b));
}

POPNET_INLINE static void compute_contiguous_strides(const popnet_shape& shape, size_t (&strides)[POPNET_MAX_DIMS])
{
  size_t stride = 1;

  for (size_t i = POPNET_MAX_DIMS; i-- > 0;)
  {
    strides[i] = stride;
    stride *= shape.dim[i];
  }
}

POPNET_INLINE static void apply_broadcast_strides(const popnet_shape& shape, size_t (&strides)[POPNET_MAX_DIMS])
{
  for (size_t i = 0; i < POPNET_MAX_DIMS; ++i)
  {
    if (shape.dim[i] == 1) strides[i] = 0;
  }
}

template <auto op, popnet_type dtype>
POPNET_NO_INLINE void binary_kernel_broadcast(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b,
                                              void* POPNET_RESTRICT c, const popnet_shape dst_shape,
                                              const popnet_shape lhs_shape, const popnet_shape rhs_shape) noexcept
{
  using traits = traits<dtype>;
  using scalar_t = typename traits::scalar_t;

  size_t lhs_strides[POPNET_MAX_DIMS];
  size_t rhs_strides[POPNET_MAX_DIMS];
  size_t dst_strides[POPNET_MAX_DIMS];

  compute_contiguous_strides(lhs_shape, lhs_strides);
  compute_contiguous_strides(rhs_shape, rhs_strides);
  compute_contiguous_strides(dst_shape, dst_strides);

  apply_broadcast_strides(lhs_shape, lhs_strides);
  apply_broadcast_strides(rhs_shape, rhs_strides);

  const bool lhs_broadcast_inner = lhs_strides[3] == 0;
  const bool rhs_broadcast_inner = rhs_strides[3] == 0;

  const auto* lhs_base = static_cast<const scalar_t*>(a);
  const auto* rhs_base = static_cast<const scalar_t*>(b);
  auto* dst_base = static_cast<scalar_t*>(c);

  const size_t inner = dst_shape.dim[3];

  for (size_t d0 = 0; d0 < dst_shape.dim[0]; ++d0)
  {
    const auto* lhs_d0 = lhs_base + d0 * lhs_strides[0];
    const auto* rhs_d0 = rhs_base + d0 * rhs_strides[0];
    auto* dst_d0 = dst_base + d0 * dst_strides[0];

    for (size_t d1 = 0; d1 < dst_shape.dim[1]; ++d1)
    {
      const auto* lhs_d1 = lhs_d0 + d1 * lhs_strides[1];
      const auto* rhs_d1 = rhs_d0 + d1 * rhs_strides[1];
      auto* dst_d1 = dst_d0 + d1 * dst_strides[1];

      for (size_t d2 = 0; d2 < dst_shape.dim[2]; ++d2)
      {
        const auto* lhs_d2 = lhs_d1 + d2 * lhs_strides[2];
        const auto* rhs_d2 = rhs_d1 + d2 * rhs_strides[2];
        auto* dst_d2 = dst_d1 + d2 * dst_strides[2];

        if (!lhs_broadcast_inner && !rhs_broadcast_inner)
        {
          simd_map<op, dtype>(dst_d2, inner, lhs_d2, rhs_d2);
        }
        else if (lhs_broadcast_inner && !rhs_broadcast_inner)
        {
          simd_map<op, dtype>(dst_d2, inner, static_cast<const broadcast*>(static_cast<const void*>(lhs_d2)), rhs_d2);
        }
        else if (!lhs_broadcast_inner && rhs_broadcast_inner)
        {
          simd_map<op, dtype>(dst_d2, inner, lhs_d2, static_cast<const broadcast*>(static_cast<const void*>(rhs_d2)));
        }
        else
        {
          simd_map<op, dtype>(dst_d2, inner, static_cast<const broadcast*>(static_cast<const void*>(lhs_d2)),
                              static_cast<const broadcast*>(static_cast<const void*>(rhs_d2)));
        }
      }
    }
  }
}

template <popnet_op op> POPNET_INLINE static void popnet_compute_unary(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];
  const size_t n = popnet_numel(dst->shape);

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

  if (src0->dtype == POPNET_TYPE_F32 && src1->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    if (src0->shape == src1->shape)
    {
      if constexpr (op == POPNET_OP_ADD) //
        binary_kernel_matching<vadd_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else if constexpr (op == POPNET_OP_SUB)
        binary_kernel_matching<vsub_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else if constexpr (op == POPNET_OP_MUL)
        binary_kernel_matching<vmul_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else if constexpr (op == POPNET_OP_DIV)
        binary_kernel_matching<vdiv_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else
        static_assert([]() { return false; }, "Not implemented");
    }
    else if (src0->shape == popnet_shape{{1, 1, 1, 1}})
    {
      if constexpr (op == POPNET_OP_ADD) //
        binary_kernel_lscalar<vadd_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else if constexpr (op == POPNET_OP_SUB)
        binary_kernel_lscalar<vsub_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else if constexpr (op == POPNET_OP_MUL)
        binary_kernel_lscalar<vmul_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else if constexpr (op == POPNET_OP_DIV)
        binary_kernel_lscalar<vdiv_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else
        static_assert([]() { return false; }, "Not implemented");
    }
    else if (src1->shape == popnet_shape{{1, 1, 1, 1}})
    {
      if constexpr (op == POPNET_OP_ADD) //
        binary_kernel_rscalar<vadd_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else if constexpr (op == POPNET_OP_SUB)
        binary_kernel_rscalar<vsub_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else if constexpr (op == POPNET_OP_MUL)
        binary_kernel_rscalar<vmul_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else if constexpr (op == POPNET_OP_DIV)
        binary_kernel_rscalar<vdiv_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, popnet_numel(dst->shape));
      else
        static_assert([]() { return false; }, "Not implemented");
    }
    else
    {
      if constexpr (op == POPNET_OP_ADD)
        binary_kernel_broadcast<vadd_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, dst->shape, src0->shape,
                                                           src1->shape);
      else if constexpr (op == POPNET_OP_SUB)
        binary_kernel_broadcast<vsub_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, dst->shape, src0->shape,
                                                           src1->shape);
      else if constexpr (op == POPNET_OP_MUL)
        binary_kernel_broadcast<vmul_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, dst->shape, src0->shape,
                                                           src1->shape);
      else if constexpr (op == POPNET_OP_DIV)
        binary_kernel_broadcast<vdiv_f32, POPNET_TYPE_F32>(src0->data, src1->data, dst->data, dst->shape, src0->shape,
                                                           src1->shape);
      else
        static_assert([]() { return false; }, "Not implemented");
    }
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
