#include "elemwise.h"

#include "vector_types.h"

#include <immintrin.h>
#include <intrin.h>

///
/// Vector Operations
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
/// Templated Kernels
///

template <auto OpFunc> void popnet_elemwise(const vf32* POPNET_RESTRICT a, vf32* POPNET_RESTRICT b, size_t n)
{

  const size_t full_vectors = n / vf32::elements;
  const size_t tail_elements = n % vf32::elements;

  for (size_t i = 0; i < full_vectors; i++)
  {
    const auto aa = vf32::load(a + i);
    const vf32 bb = OpFunc(aa);
    bb.store(b + i);
  }

  if (tail_elements > 0)
  {
    const vf32::mask_t m = (1 << tail_elements) - 1;
    const auto aa = vf32::load(a + full_vectors, m);
    const vf32 bb = OpFunc(aa);
    bb.store(b + full_vectors, m);
  }
}

template <auto OpFunc>
void popnet_elemwise(const vf32* POPNET_RESTRICT a, const vf32* POPNET_RESTRICT b, vf32* POPNET_RESTRICT c, size_t n)
{

  const size_t full_vectors = n / vf32::elements;
  const size_t tail_elements = n % vf32::elements;

  for (size_t i = 0; i < full_vectors; i++)
  {
    const auto [aa, bb] = vf32::load2(a + i, b + i);
    const vf32 cc = OpFunc(aa, bb);
    cc.store(c + i);
  }

  if (tail_elements > 0)
  {
    const vf32::mask_t m = (1 << tail_elements) - 1;
    const auto [aa, bb] = vf32::load2(a + full_vectors, b + full_vectors, m);
    const vf32 cc = OpFunc(aa, bb);
    cc.store(c + full_vectors, m);
  }
}

///
/// Kernels
///

void popnet_add_f32(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b, void* POPNET_RESTRICT c, size_t n)
{
  return popnet_elemwise<op_add>(static_cast<const vf32*>(a), static_cast<const vf32*>(b), static_cast<vf32*>(c), n);
}

void popnet_sub_f32(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b, void* POPNET_RESTRICT c, size_t n)
{
  return popnet_elemwise<op_sub>(static_cast<const vf32*>(a), static_cast<const vf32*>(b), static_cast<vf32*>(c), n);
}

void popnet_mul_f32(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b, void* POPNET_RESTRICT c, size_t n)
{
  return popnet_elemwise<op_mul>(static_cast<const vf32*>(a), static_cast<const vf32*>(b), static_cast<vf32*>(c), n);
}

void popnet_div_f32(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b, void* POPNET_RESTRICT c, size_t n)
{
  return popnet_elemwise<op_div>(static_cast<const vf32*>(a), static_cast<const vf32*>(b), static_cast<vf32*>(c), n);
}

void popnet_abs_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_abs>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_neg_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_neg>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_sqr_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_sqr>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_sqrt_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_sqrt>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_exp_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_exp>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_log_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_log>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_sin_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_sin>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_cos_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_cos>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_tanh_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_tanh>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_relu_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_relu>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}

void popnet_sigmoid_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n)
{
  return popnet_elemwise<op_sigmoid>(static_cast<const vf32*>(a), static_cast<vf32*>(b), n);
}
