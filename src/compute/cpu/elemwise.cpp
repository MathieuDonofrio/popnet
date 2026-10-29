#include "elemwise.h"

#include "dtype_traits.h"
#include "simd_algorithms.h"
#include "tensor.h"

template <auto op, popnet_dtype dtype>
POPNET_NO_INLINE void unary_kernel(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n) noexcept
{
  POPNET_EXPECT(n > 0);
  simd_map<op, dtype>(b, n, a);
}

template <auto op, popnet_dtype dtype>
POPNET_NO_INLINE void binary_kernel_matching(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b,
                                             void* POPNET_RESTRICT c, size_t n) noexcept
{
  POPNET_EXPECT(n > 0);
  simd_map<op, dtype>(c, n, a, b);
}

template <auto op, popnet_dtype dtype>
POPNET_NO_INLINE void binary_kernel_lscalar(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b,
                                            void* POPNET_RESTRICT c, size_t n) noexcept
{
  POPNET_EXPECT(n > 0);
  simd_map<op, dtype>(c, n, static_cast<const broadcast*>(a), b);
}

template <auto op, popnet_dtype dtype>
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

template <auto op, popnet_dtype dtype>
POPNET_NO_INLINE void binary_kernel_broadcast(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b,
                                              void* POPNET_RESTRICT c, const popnet_shape dst_shape,
                                              const popnet_shape lhs_shape, const popnet_shape rhs_shape) noexcept
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = typename dtype_traits::scalar_t;

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

  if (src->dtype == popnet_dtype_f32 && dst->dtype == popnet_dtype_f32)
  {
    if constexpr (op == popnet_op_abs) unary_kernel<simd::abs_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_neg)
      unary_kernel<simd::neg_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_sqr)
      unary_kernel<simd::sqr_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_sqrt)
      unary_kernel<simd::sqrt_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_exp)
      unary_kernel<simd::exp_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_log)
      unary_kernel<simd::log_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_sin)
      unary_kernel<simd::sin_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_cos)
      unary_kernel<simd::cos_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_tanh)
      unary_kernel<simd::tanh_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_relu)
      unary_kernel<simd::relu_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else if constexpr (op == popnet_op_sigmoid)
      unary_kernel<simd::sigmoid_f32, popnet_dtype_f32>(src->data, dst->data, n);
    else
      POPNET_UNSUPPORTED_STATIC("Unsupported elementwise operation");
  }
  else
  {
    POPNET_UNSUPPORTED("Elementwise operations require f32 inputs and output");
  }
}

template <auto op> POPNET_INLINE static void popnet_compute_binary(popnet_tensor* dst)
{
  const popnet_tensor* src0 = dst->src[0];
  const popnet_tensor* src1 = dst->src[1];

  if (src0->dtype == popnet_dtype_f32 && src1->dtype == popnet_dtype_f32 && dst->dtype == popnet_dtype_f32)
  {
    if (src0->shape == src1->shape)
    {
      if constexpr (op == popnet_op_add)
        binary_kernel_matching<simd::add_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                                popnet_numel(dst->shape));
      else if constexpr (op == popnet_op_sub)
        binary_kernel_matching<simd::sub_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                                popnet_numel(dst->shape));
      else if constexpr (op == popnet_op_mul)
        binary_kernel_matching<simd::mul_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                                popnet_numel(dst->shape));
      else if constexpr (op == popnet_op_div)
        binary_kernel_matching<simd::div_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                                popnet_numel(dst->shape));
      else
        POPNET_UNSUPPORTED_STATIC("Unsupported elementwise operation");
    }
    else if (src0->shape == popnet_scalar_shape)
    {
      if constexpr (op == popnet_op_add)
        binary_kernel_lscalar<simd::add_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                               popnet_numel(dst->shape));
      else if constexpr (op == popnet_op_sub)
        binary_kernel_lscalar<simd::sub_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                               popnet_numel(dst->shape));
      else if constexpr (op == popnet_op_mul)
        binary_kernel_lscalar<simd::mul_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                               popnet_numel(dst->shape));
      else if constexpr (op == popnet_op_div)
        binary_kernel_lscalar<simd::div_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                               popnet_numel(dst->shape));
      else
        POPNET_UNSUPPORTED_STATIC("Unsupported elementwise operation");
    }
    else if (src1->shape == popnet_scalar_shape)
    {
      if constexpr (op == popnet_op_add)
        binary_kernel_rscalar<simd::add_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                               popnet_numel(dst->shape));
      else if constexpr (op == popnet_op_sub)
        binary_kernel_rscalar<simd::sub_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                               popnet_numel(dst->shape));
      else if constexpr (op == popnet_op_mul)
        binary_kernel_rscalar<simd::mul_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                               popnet_numel(dst->shape));
      else if constexpr (op == popnet_op_div)
        binary_kernel_rscalar<simd::div_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data,
                                                               popnet_numel(dst->shape));
      else
        POPNET_UNSUPPORTED_STATIC("Unsupported elementwise operation");
    }
    else
    {
      if constexpr (op == popnet_op_add)
        binary_kernel_broadcast<simd::add_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data, dst->shape,
                                                                 src0->shape, src1->shape);
      else if constexpr (op == popnet_op_sub)
        binary_kernel_broadcast<simd::sub_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data, dst->shape,
                                                                 src0->shape, src1->shape);
      else if constexpr (op == popnet_op_mul)
        binary_kernel_broadcast<simd::mul_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data, dst->shape,
                                                                 src0->shape, src1->shape);
      else if constexpr (op == popnet_op_div)
        binary_kernel_broadcast<simd::div_f32, popnet_dtype_f32>(src0->data, src1->data, dst->data, dst->shape,
                                                                 src0->shape, src1->shape);
      else
        POPNET_UNSUPPORTED_STATIC("Unsupported elementwise operation");
    }
  }
  else
  {
    POPNET_UNSUPPORTED("Elementwise operations require f32 inputs and output");
  }
}

void popnet_compute_add(popnet_tensor* dst) { popnet_compute_binary<popnet_op_add>(dst); }
void popnet_compute_sub(popnet_tensor* dst) { popnet_compute_binary<popnet_op_sub>(dst); }
void popnet_compute_mul(popnet_tensor* dst) { popnet_compute_binary<popnet_op_mul>(dst); }
void popnet_compute_div(popnet_tensor* dst) { popnet_compute_binary<popnet_op_div>(dst); }
void popnet_compute_abs(popnet_tensor* dst) { popnet_compute_unary<popnet_op_abs>(dst); }
void popnet_compute_neg(popnet_tensor* dst) { popnet_compute_unary<popnet_op_neg>(dst); }
void popnet_compute_sqr(popnet_tensor* dst) { popnet_compute_unary<popnet_op_sqr>(dst); }
void popnet_compute_sqrt(popnet_tensor* dst) { popnet_compute_unary<popnet_op_sqrt>(dst); }
void popnet_compute_exp(popnet_tensor* dst) { popnet_compute_unary<popnet_op_exp>(dst); }
void popnet_compute_log(popnet_tensor* dst) { popnet_compute_unary<popnet_op_log>(dst); }
void popnet_compute_sin(popnet_tensor* dst) { popnet_compute_unary<popnet_op_sin>(dst); }
void popnet_compute_cos(popnet_tensor* dst) { popnet_compute_unary<popnet_op_cos>(dst); }
void popnet_compute_tanh(popnet_tensor* dst) { popnet_compute_unary<popnet_op_tanh>(dst); }
void popnet_compute_relu(popnet_tensor* dst) { popnet_compute_unary<popnet_op_relu>(dst); }
void popnet_compute_sigmoid(popnet_tensor* dst) { popnet_compute_unary<popnet_op_sigmoid>(dst); }
