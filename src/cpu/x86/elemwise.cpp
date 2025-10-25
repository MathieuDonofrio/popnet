#include "elemwise.h"

#include "tensor.h"
#include "vector_types.h"

#include <immintrin.h>
#include <intrin.h>

///
/// Fixed-Length Operations
///

POPNET_INLINE static vf32 op_add(vf32 a, vf32 b) { return vf32{_mm512_add_ps(a.v, b.v)}; }
POPNET_INLINE static vf32 op_sub(vf32 a, vf32 b) { return vf32{_mm512_sub_ps(a.v, b.v)}; }
POPNET_INLINE static vf32 op_mul(vf32 a, vf32 b) { return vf32{_mm512_mul_ps(a.v, b.v)}; }
POPNET_INLINE static vf32 op_div(vf32 a, vf32 b) { return vf32{_mm512_div_ps(a.v, b.v)}; }

POPNET_INLINE static vf32 op_abs(vf32 a) { return vf32{_mm512_abs_ps(a.v)}; }
POPNET_INLINE static vf32 op_neg(vf32 a) { return vf32{_mm512_sub_ps(_mm512_setzero_ps(), a.v)}; }

POPNET_INLINE static vf32 op_sqr(vf32 a) { return vf32{_mm512_mul_ps(a.v, a.v)}; }
POPNET_INLINE static vf32 op_sqrt(vf32 a) { return vf32{_mm512_sqrt_ps(a.v)}; }

POPNET_INLINE static vf32 op_exp(vf32 a) { return vf32{_mm512_exp_ps(a.v)}; }
POPNET_INLINE static vf32 op_log(vf32 a) { return vf32{_mm512_log_ps(a.v)}; }

POPNET_INLINE static vf32 op_sin(vf32 a) { return vf32{_mm512_sin_ps(a.v)}; }
POPNET_INLINE static vf32 op_cos(vf32 a) { return vf32{_mm512_cos_ps(a.v)}; }
POPNET_INLINE static vf32 op_tanh(vf32 a) { return vf32{_mm512_tanh_ps(a.v)}; }

POPNET_INLINE static vf32 op_relu(vf32 a) { return vf32{_mm512_max_ps(a.v, _mm512_setzero_ps())}; }

POPNET_INLINE static vf32 op_sigmoid(vf32 a)
{
  const auto vneg = _mm512_sub_ps(_mm512_setzero_ps(), a.v);
  const auto vnegexp = _mm512_exp_ps(vneg);
  const auto vdenom = _mm512_add_ps(_mm512_set1_ps(1.0f), vnegexp);
  return vf32{_mm512_rcp14_ps(vdenom)}; // Approximation
}

///
/// Variable-Length Operations
///

template <auto UnaryOpFunc>
POPNET_INLINE static void vec_op_unary(const vf32* POPNET_RESTRICT a, vf32* POPNET_RESTRICT b, size_t n)
{
  const size_t full_vectors = n / vf32::elements;
  const size_t tail_elements = n % vf32::elements;

  for (size_t i = 0; i < full_vectors; i++)
  {
    const auto aa = vf32::load(a + i);
    const vf32 bb = UnaryOpFunc(aa);
    bb.store(b + i);
  }

  if (tail_elements > 0)
  {
    const vf32::mask_t m = (1 << tail_elements) - 1;
    const auto aa = vf32::load(a + full_vectors, m);
    const vf32 bb = UnaryOpFunc(aa);
    bb.store(b + full_vectors, m);
  }
}

template <auto BinaryOpFunc>
POPNET_INLINE static void vec_op_binary(const vf32* POPNET_RESTRICT a, const vf32* POPNET_RESTRICT b,
                                        vf32* POPNET_RESTRICT c, size_t n)
{
  const size_t full_vectors = n / vf32::elements;
  const size_t tail_elements = n % vf32::elements;

  for (size_t i = 0; i < full_vectors; i++)
  {
    const auto [aa, bb] = vf32::load2(a + i, b + i);
    const vf32 cc = BinaryOpFunc(aa, bb);
    cc.store(c + i);
  }

  if (tail_elements > 0)
  {
    const vf32::mask_t m = (1 << tail_elements) - 1;
    const auto [aa, bb] = vf32::load2(a + full_vectors, b + full_vectors, m);
    const vf32 cc = BinaryOpFunc(aa, bb);
    cc.store(c + full_vectors, m);
  }
}

///
/// Templated Kernels
///

template <auto UnaryOpFunc> POPNET_INLINE static void popnet_compute_unary(popnet_tensor* dst)
{
  const popnet_tensor* src = dst->src[0];
  const size_t n = popnet_nelems(dst->shape);

  if (src->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    vec_op_unary<UnaryOpFunc>(static_cast<const vf32*>(src->data), static_cast<vf32*>(dst->data), n);
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

  if (src1->dtype == POPNET_TYPE_F32 && dst->dtype == POPNET_TYPE_F32)
  {
    vec_op_binary<BinaryOpFunc>(static_cast<const vf32*>(src0->data), static_cast<const vf32*>(src1->data),
                                static_cast<vf32*>(dst->data), n);
  }
  else
  {
    POPNET_UNREACHABLE(); // TODO
  }
}

///
/// Kernels
///

void popnet_compute_add(popnet_tensor* dst) { popnet_compute_binary<op_add>(dst); }
void popnet_compute_sub(popnet_tensor* dst) { popnet_compute_binary<op_sub>(dst); }
void popnet_compute_mul(popnet_tensor* dst) { popnet_compute_binary<op_mul>(dst); }
void popnet_compute_div(popnet_tensor* dst) { popnet_compute_binary<op_div>(dst); }
void popnet_compute_abs(popnet_tensor* dst) { popnet_compute_unary<op_abs>(dst); }
void popnet_compute_neg(popnet_tensor* dst) { popnet_compute_unary<op_neg>(dst); }
void popnet_compute_sqr(popnet_tensor* dst) { popnet_compute_unary<op_sqr>(dst); }
void popnet_compute_sqrt(popnet_tensor* dst) { popnet_compute_unary<op_sqrt>(dst); }
void popnet_compute_exp(popnet_tensor* dst) { popnet_compute_unary<op_exp>(dst); }
void popnet_compute_log(popnet_tensor* dst) { popnet_compute_unary<op_log>(dst); }
void popnet_compute_sin(popnet_tensor* dst) { popnet_compute_unary<op_sin>(dst); }
void popnet_compute_cos(popnet_tensor* dst) { popnet_compute_unary<op_cos>(dst); }
void popnet_compute_tanh(popnet_tensor* dst) { popnet_compute_unary<op_tanh>(dst); }
void popnet_compute_relu(popnet_tensor* dst) { popnet_compute_unary<op_relu>(dst); }
void popnet_compute_sigmoid(popnet_tensor* dst) { popnet_compute_unary<op_sigmoid>(dst); }
