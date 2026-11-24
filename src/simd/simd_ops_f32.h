#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "simd_traits.h"

// Avoid ARM NEON names: MSVC defines some intrinsics as macros.
namespace simd
{

#if defined(POPNET_SVE)
// SVE permits up to 2048 bits per vector. Only the runtime lane count is used.
inline constexpr size_t sve_max_f32_lanes = 2048 / 32;
#endif

#if !defined(POPNET_SCALAR) && !defined(POPNET_AVX512)

template <typename Op> POPNET_INLINE vf32_t map_f32(vf32_t a, Op op)
{
  using traits = simd_traits<popnet_dtype_f32>;
#if defined(POPNET_SVE)
  f32_t lanes[sve_max_f32_lanes];
#else
  f32_t lanes[traits::vector_length()];
#endif
  traits::vstore(lanes, a);

  const size_t n = traits::vector_length();
  for (size_t i = 0; i < n; ++i)
    lanes[i] = op(lanes[i]);

  return traits::vload(lanes);
}

POPNET_INLINE vf32_t exp_f32(vf32_t a) { return map_f32(a, [](float x) { return std::exp(x); }); }
POPNET_INLINE vf32_t log_f32(vf32_t a) { return map_f32(a, [](float x) { return std::log(x); }); }
POPNET_INLINE vf32_t sin_f32(vf32_t a) { return map_f32(a, [](float x) { return std::sin(x); }); }
POPNET_INLINE vf32_t cos_f32(vf32_t a) { return map_f32(a, [](float x) { return std::cos(x); }); }
POPNET_INLINE vf32_t tanh_f32(vf32_t a) { return map_f32(a, [](float x) { return std::tanh(x); }); }

POPNET_INLINE vf32_t sigmoid_f32(vf32_t a)
{
  return map_f32(a, [](float x) {
    const auto e = std::exp(-std::abs(x));
    return (x < 0 ? e : 1.0f) / (1.0f + e);
  });
}

#endif

#if defined(POPNET_AVX512)

POPNET_INLINE vf32_t set_zero_f32() { return _mm512_setzero_ps(); }
POPNET_INLINE vf32_t set_one_f32() { return _mm512_set1_ps(1.0f); }
POPNET_INLINE vf32_t set_neg_inf_f32() { return _mm512_set1_ps(-std::numeric_limits<f32_t>::infinity()); }
POPNET_INLINE vf32_t set_pos_inf_f32() { return _mm512_set1_ps(std::numeric_limits<f32_t>::infinity()); }

POPNET_INLINE vf32_t abs_f32(vf32_t a) { return _mm512_abs_ps(a); }
POPNET_INLINE vf32_t neg_f32(vf32_t a)
{
  return _mm512_castsi512_ps(
      _mm512_xor_si512(_mm512_castps_si512(a), _mm512_set1_epi32(std::numeric_limits<int32_t>::min())));
}

POPNET_INLINE vf32_t sqr_f32(vf32_t a) { return _mm512_mul_ps(a, a); }
POPNET_INLINE vf32_t sqrt_f32(vf32_t a) { return _mm512_sqrt_ps(a); }

#if defined(_MSC_VER) && !defined(__clang__)
POPNET_INLINE vf32_t exp_f32(vf32_t a) { return _mm512_exp_ps(a); }
POPNET_INLINE vf32_t log_f32(vf32_t a) { return _mm512_log_ps(a); }

POPNET_INLINE vf32_t sin_f32(vf32_t a) { return _mm512_sin_ps(a); }
POPNET_INLINE vf32_t cos_f32(vf32_t a) { return _mm512_cos_ps(a); }
POPNET_INLINE vf32_t tanh_f32(vf32_t a) { return _mm512_tanh_ps(a); }
#else
POPNET_INLINE vf32_t exp_f32(vf32_t a)
{
  const auto zero = _mm512_setzero_ps();
  const auto upper = _mm512_set1_ps(88.722839f);
  const auto lower = _mm512_set1_ps(-103.972084f);

  const auto nan = _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q);
  const auto high = _mm512_cmp_ps_mask(a, upper, _CMP_GT_OQ);
  const auto low = _mm512_cmp_ps_mask(a, lower, _CMP_LT_OQ);

  const auto x = _mm512_max_ps(lower, _mm512_min_ps(upper, a));

  // exp(x) = 2^n * exp(r).
  const auto n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  auto r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693145751953125f), x);
  r = _mm512_fnmadd_ps(n, _mm512_set1_ps(1.428606765330187e-6f), r);

  // Degree-8 exp polynomial.
  auto p = _mm512_set1_ps(1.0f / 40320.0f);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 5040.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 720.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));

  auto y = _mm512_scalef_ps(p, n);

  y = _mm512_mask_mov_ps(y, high, _mm512_set1_ps(std::numeric_limits<float>::infinity()));
  y = _mm512_mask_mov_ps(y, low, zero);

  return _mm512_mask_mov_ps(y, nan, a);
}

POPNET_INLINE vf32_t log_f32(vf32_t a)
{
  const auto zero = _mm512_setzero_ps();
  const auto one = _mm512_set1_ps(1.0f);
  const auto inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  const auto valid = _mm512_cmp_ps_mask(a, zero, _CMP_GT_OQ) & _mm512_cmp_ps_mask(a, inf, _CMP_LT_OQ);

  const auto x = _mm512_mask_mov_ps(one, valid, a);

  // x = m * 2^e, with m in [0.75, 1.5).
  const auto m = _mm512_getmant_ps(x, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_zero);

  auto e = _mm512_getexp_ps(x);
  e = _mm512_mask_add_ps(e, _mm512_cmp_ps_mask(m, one, _CMP_LT_OQ), e, one);

  // log(m) = 2 * atanh((m-1)/(m+1)).
  const auto t = _mm512_div_ps(_mm512_sub_ps(m, one), _mm512_add_ps(m, one));

  const auto z = _mm512_mul_ps(t, t);

  auto p = _mm512_set1_ps(1.0f / 13.0f);
  p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(1.0f / 11.0f));
  p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(1.0f / 9.0f));
  p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(1.0f / 7.0f));
  p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(1.0f / 5.0f));
  p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(1.0f / 3.0f));
  p = _mm512_fmadd_ps(p, z, one);

  const auto lm = _mm512_mul_ps(_mm512_add_ps(t, t), p);

  auto y = _mm512_fmadd_ps(e, _mm512_set1_ps(0.6931471805599453f), lm);

  const auto negative = _mm512_cmp_ps_mask(a, zero, _CMP_LT_OQ);
  const auto is_zero = _mm512_cmp_ps_mask(a, zero, _CMP_EQ_OQ);
  const auto is_inf = _mm512_cmp_ps_mask(a, inf, _CMP_EQ_OQ);
  const auto is_nan = _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q);

  y = _mm512_mask_mov_ps(y, negative, _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()));
  y = _mm512_mask_mov_ps(y, is_zero, _mm512_set1_ps(-std::numeric_limits<float>::infinity()));
  y = _mm512_mask_mov_ps(y, is_inf, inf);

  return _mm512_mask_mov_ps(y, is_nan, a);
}

template <typename Op> POPNET_INLINE vf32_t map_f32(vf32_t a, Op op)
{
  alignas(64) float lanes[16];

  _mm512_store_ps(lanes, a);

  for (auto& lane : lanes)
    lane = op(lane);

  return _mm512_load_ps(lanes);
}

POPNET_INLINE vf32_t sin_f32(vf32_t a)
{
  return map_f32(a, [](float x) { return std::sin(x); });
}
POPNET_INLINE vf32_t cos_f32(vf32_t a)
{
  return map_f32(a, [](float x) { return std::cos(x); });
}

POPNET_INLINE vf32_t tanh_f32(vf32_t a)
{
  const auto one = _mm512_set1_ps(1.0f);

  const auto sign = _mm512_and_si512(_mm512_castps_si512(a), _mm512_set1_epi32(std::numeric_limits<int32_t>::min()));

  const auto ax = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_set1_epi32(0x7fffffff)));

  const auto small = _mm512_cmp_ps_mask(ax, _mm512_set1_ps(2.5f), _CMP_LE_OQ);

  const auto x = _mm512_maskz_mov_ps(small, ax);
  const auto z = _mm512_mul_ps(x, x);

  // Rational approximation near zero.
  auto n = _mm512_fmadd_ps(z, one, _mm512_set1_ps(378.0f));
  n = _mm512_fmadd_ps(n, z, _mm512_set1_ps(17325.0f));
  n = _mm512_fmadd_ps(n, z, _mm512_set1_ps(135135.0f));

  auto d = _mm512_fmadd_ps(z, _mm512_set1_ps(28.0f), _mm512_set1_ps(3150.0f));
  d = _mm512_fmadd_ps(d, z, _mm512_set1_ps(62370.0f));
  d = _mm512_fmadd_ps(d, z, _mm512_set1_ps(135135.0f));

  const auto near = _mm512_div_ps(_mm512_mul_ps(x, n), d);

  // Stable exponential form for larger arguments.
  // Naturally saturates to 1 for very large |x|.
  const auto e = exp_f32(_mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(ax, _mm512_set1_ps(2.0f))));

  const auto far = _mm512_div_ps(_mm512_sub_ps(one, e), _mm512_add_ps(one, e));

  auto y = _mm512_mask_blend_ps(small, far, near);

  // Restore the sign, including negative zero.
  y = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(y), sign));

  const auto nan = _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q);

  return _mm512_mask_mov_ps(y, nan, a);
}

#endif

POPNET_INLINE vf32_t relu_f32(vf32_t a) { return _mm512_max_ps(a, _mm512_setzero_ps()); }

POPNET_INLINE vf32_t sigmoid_f32(vf32_t a)
{
  const auto denominator = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_f32(neg_f32(a)));
  // One Newton step reduces the estimate's relative error from about 2^-14
  // to F32 rounding error, well below the sigmoid tolerance.
  const auto estimate = _mm512_rcp14_ps(denominator);
  const auto correction = _mm512_fnmadd_ps(denominator, estimate, _mm512_set1_ps(2.0f));
  // Overflow of exp(-x) means sigmoid(x) is far below the absolute tolerance.
  // Mask that result to zero; leave NaNs active so they propagate.
  const auto not_overflow = _mm512_cmp_ps_mask(denominator, set_pos_inf_f32(), _CMP_NEQ_UQ);
  return _mm512_maskz_mul_ps(not_overflow, estimate, correction);
}

POPNET_INLINE vf32_t add_f32(vf32_t a, vf32_t b) { return _mm512_add_ps(a, b); }
POPNET_INLINE vf32_t sub_f32(vf32_t a, vf32_t b) { return _mm512_sub_ps(a, b); }
POPNET_INLINE vf32_t mul_f32(vf32_t a, vf32_t b) { return _mm512_mul_ps(a, b); }
POPNET_INLINE vf32_t div_f32(vf32_t a, vf32_t b) { return _mm512_div_ps(a, b); }
POPNET_INLINE vf32_t min_f32(vf32_t a, vf32_t b) { return _mm512_min_ps(a, b); }
POPNET_INLINE vf32_t max_f32(vf32_t a, vf32_t b) { return _mm512_max_ps(a, b); }

POPNET_INLINE vf32_t fma_f32(vf32_t a, vf32_t b, vf32_t c) { return _mm512_fmadd_ps(a, b, c); }

POPNET_INLINE f32_t reduce_sum_f32(vf32_t a) { return _mm512_reduce_add_ps(a); }
POPNET_INLINE f32_t reduce_product_f32(vf32_t a) { return _mm512_reduce_mul_ps(a); }
POPNET_INLINE f32_t reduce_min_f32(vf32_t a) { return _mm512_reduce_min_ps(a); }
POPNET_INLINE f32_t reduce_max_f32(vf32_t a) { return _mm512_reduce_max_ps(a); }

#elif defined(POPNET_AVX) || defined(POPNET_SSE2)

#if defined(POPNET_AVX)
#define POPNET_F32_INTRINSIC(op) _mm256_##op
#else
#define POPNET_F32_INTRINSIC(op) _mm_##op
#endif

POPNET_INLINE vf32_t set_zero_f32() { return POPNET_F32_INTRINSIC(setzero_ps)(); }
POPNET_INLINE vf32_t set_one_f32() { return POPNET_F32_INTRINSIC(set1_ps)(1.0f); }
POPNET_INLINE vf32_t set_neg_inf_f32()
{
  return POPNET_F32_INTRINSIC(set1_ps)(-std::numeric_limits<f32_t>::infinity());
}
POPNET_INLINE vf32_t set_pos_inf_f32()
{
  return POPNET_F32_INTRINSIC(set1_ps)(std::numeric_limits<f32_t>::infinity());
}

POPNET_INLINE vf32_t abs_f32(vf32_t a)
{
  return POPNET_F32_INTRINSIC(andnot_ps)(POPNET_F32_INTRINSIC(set1_ps)(-0.0f), a);
}
POPNET_INLINE vf32_t neg_f32(vf32_t a)
{
  return POPNET_F32_INTRINSIC(xor_ps)(a, POPNET_F32_INTRINSIC(set1_ps)(-0.0f));
}

POPNET_INLINE vf32_t sqr_f32(vf32_t a) { return POPNET_F32_INTRINSIC(mul_ps)(a, a); }
POPNET_INLINE vf32_t sqrt_f32(vf32_t a) { return POPNET_F32_INTRINSIC(sqrt_ps)(a); }

POPNET_INLINE vf32_t relu_f32(vf32_t a) { return POPNET_F32_INTRINSIC(max_ps)(set_zero_f32(), a); }

POPNET_INLINE vf32_t add_f32(vf32_t a, vf32_t b) { return POPNET_F32_INTRINSIC(add_ps)(a, b); }
POPNET_INLINE vf32_t sub_f32(vf32_t a, vf32_t b) { return POPNET_F32_INTRINSIC(sub_ps)(a, b); }
POPNET_INLINE vf32_t mul_f32(vf32_t a, vf32_t b) { return POPNET_F32_INTRINSIC(mul_ps)(a, b); }
POPNET_INLINE vf32_t div_f32(vf32_t a, vf32_t b) { return POPNET_F32_INTRINSIC(div_ps)(a, b); }
POPNET_INLINE vf32_t min_f32(vf32_t a, vf32_t b) { return POPNET_F32_INTRINSIC(min_ps)(b, a); }
POPNET_INLINE vf32_t max_f32(vf32_t a, vf32_t b) { return POPNET_F32_INTRINSIC(max_ps)(b, a); }

POPNET_INLINE vf32_t fma_f32(vf32_t a, vf32_t b, vf32_t c)
{
  // MSVC's AVX2 target includes FMA. Clang and GCC report it separately.
#if defined(POPNET_AVX) && (defined(__FMA__) || (defined(_MSC_VER) && !defined(__clang__) && defined(__AVX2__)))
  return _mm256_fmadd_ps(a, b, c);
#else
  return add_f32(mul_f32(a, b), c);
#endif
}

template <auto op> POPNET_INLINE f32_t reduce_lanes_f32(vf32_t a)
{
#if defined(POPNET_AVX)
  a = op(a, _mm256_permute2f128_ps(a, a, 1));
  a = op(a, _mm256_permute_ps(a, _MM_SHUFFLE(1, 0, 3, 2)));
  a = op(a, _mm256_permute_ps(a, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtss_f32(_mm256_castps256_ps128(a));
#else
  a = op(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 0, 3, 2)));
  a = op(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtss_f32(a);
#endif
}

POPNET_INLINE f32_t reduce_sum_f32(vf32_t a) { return reduce_lanes_f32<add_f32>(a); }
POPNET_INLINE f32_t reduce_product_f32(vf32_t a) { return reduce_lanes_f32<mul_f32>(a); }
POPNET_INLINE f32_t reduce_min_f32(vf32_t a) { return reduce_lanes_f32<min_f32>(a); }
POPNET_INLINE f32_t reduce_max_f32(vf32_t a) { return reduce_lanes_f32<max_f32>(a); }

#undef POPNET_F32_INTRINSIC

#elif defined(POPNET_SVE)

POPNET_INLINE vf32_t set_zero_f32() { return svdup_n_f32(0.0f); }
POPNET_INLINE vf32_t set_one_f32() { return svdup_n_f32(1.0f); }
POPNET_INLINE vf32_t set_neg_inf_f32() { return svdup_n_f32(-std::numeric_limits<f32_t>::infinity()); }
POPNET_INLINE vf32_t set_pos_inf_f32() { return svdup_n_f32(std::numeric_limits<f32_t>::infinity()); }

POPNET_INLINE vf32_t abs_f32(vf32_t a) { return svabs_f32_x(svptrue_b32(), a); }
POPNET_INLINE vf32_t neg_f32(vf32_t a) { return svneg_f32_x(svptrue_b32(), a); }
POPNET_INLINE vf32_t sqr_f32(vf32_t a) { return svmul_f32_x(svptrue_b32(), a, a); }
POPNET_INLINE vf32_t sqrt_f32(vf32_t a) { return svsqrt_f32_x(svptrue_b32(), a); }

POPNET_INLINE vf32_t relu_f32(vf32_t a)
{
  return svsel_f32(svcmplt_n_f32(svptrue_b32(), a, 0.0f), set_zero_f32(), a);
}

POPNET_INLINE vf32_t add_f32(vf32_t a, vf32_t b) { return svadd_f32_x(svptrue_b32(), a, b); }
POPNET_INLINE vf32_t sub_f32(vf32_t a, vf32_t b) { return svsub_f32_x(svptrue_b32(), a, b); }
POPNET_INLINE vf32_t mul_f32(vf32_t a, vf32_t b) { return svmul_f32_x(svptrue_b32(), a, b); }
POPNET_INLINE vf32_t div_f32(vf32_t a, vf32_t b) { return svdiv_f32_x(svptrue_b32(), a, b); }
POPNET_INLINE vf32_t min_f32(vf32_t a, vf32_t b) { return svmin_f32_x(svptrue_b32(), a, b); }
POPNET_INLINE vf32_t max_f32(vf32_t a, vf32_t b) { return svmax_f32_x(svptrue_b32(), a, b); }

POPNET_INLINE vf32_t fma_f32(vf32_t a, vf32_t b, vf32_t c) { return svmla_f32_x(svptrue_b32(), c, a, b); }

POPNET_INLINE f32_t reduce_sum_f32(vf32_t a) { return svaddv_f32(svptrue_b32(), a); }
POPNET_INLINE f32_t reduce_min_f32(vf32_t a) { return svminv_f32(svptrue_b32(), a); }
POPNET_INLINE f32_t reduce_max_f32(vf32_t a) { return svmaxv_f32(svptrue_b32(), a); }

POPNET_INLINE f32_t reduce_product_f32(vf32_t a)
{
  using traits = simd_traits<popnet_dtype_f32>;
  f32_t lanes[sve_max_f32_lanes];
  traits::vstore(lanes, a);

  f32_t product = 1.0f;
  const size_t n = traits::vector_length();
  for (size_t i = 0; i < n; ++i)
    product *= lanes[i];

  return product;
}

#elif defined(POPNET_NEON)

POPNET_INLINE vf32_t set_zero_f32() { return vdupq_n_f32(0.0f); }
POPNET_INLINE vf32_t set_one_f32() { return vdupq_n_f32(1.0f); }
POPNET_INLINE vf32_t set_neg_inf_f32() { return vdupq_n_f32(-std::numeric_limits<f32_t>::infinity()); }
POPNET_INLINE vf32_t set_pos_inf_f32() { return vdupq_n_f32(std::numeric_limits<f32_t>::infinity()); }

POPNET_INLINE vf32_t abs_f32(vf32_t a) { return vabsq_f32(a); }
POPNET_INLINE vf32_t neg_f32(vf32_t a) { return vnegq_f32(a); }
POPNET_INLINE vf32_t sqr_f32(vf32_t a) { return vmulq_f32(a, a); }
POPNET_INLINE vf32_t sqrt_f32(vf32_t a) { return vsqrtq_f32(a); }

POPNET_INLINE vf32_t relu_f32(vf32_t a)
{
  const auto zero = set_zero_f32();
  return vbslq_f32(vcltq_f32(a, zero), zero, a);
}

POPNET_INLINE vf32_t add_f32(vf32_t a, vf32_t b) { return vaddq_f32(a, b); }
POPNET_INLINE vf32_t sub_f32(vf32_t a, vf32_t b) { return vsubq_f32(a, b); }
POPNET_INLINE vf32_t mul_f32(vf32_t a, vf32_t b) { return vmulq_f32(a, b); }
POPNET_INLINE vf32_t div_f32(vf32_t a, vf32_t b) { return vdivq_f32(a, b); }
POPNET_INLINE vf32_t min_f32(vf32_t a, vf32_t b) { return vminq_f32(a, b); }
POPNET_INLINE vf32_t max_f32(vf32_t a, vf32_t b) { return vmaxq_f32(a, b); }

POPNET_INLINE vf32_t fma_f32(vf32_t a, vf32_t b, vf32_t c) { return vfmaq_f32(c, a, b); }

POPNET_INLINE f32_t reduce_sum_f32(vf32_t a) { return vaddvq_f32(a); }
POPNET_INLINE f32_t reduce_min_f32(vf32_t a) { return vminvq_f32(a); }
POPNET_INLINE f32_t reduce_max_f32(vf32_t a) { return vmaxvq_f32(a); }

POPNET_INLINE f32_t reduce_product_f32(vf32_t a)
{
  const auto pairs = vmul_f32(vget_low_f32(a), vget_high_f32(a));
  return vget_lane_f32(pairs, 0) * vget_lane_f32(pairs, 1);
}

#else

POPNET_INLINE vf32_t set_zero_f32() { return 0; }
POPNET_INLINE vf32_t set_one_f32() { return 1; }
POPNET_INLINE vf32_t set_neg_inf_f32() { return -std::numeric_limits<f32_t>::infinity(); }
POPNET_INLINE vf32_t set_pos_inf_f32() { return std::numeric_limits<f32_t>::infinity(); }

POPNET_INLINE vf32_t abs_f32(vf32_t a) { return std::fabs(a); }
POPNET_INLINE vf32_t neg_f32(vf32_t a) { return -a; }

POPNET_INLINE vf32_t sqr_f32(vf32_t a) { return a * a; }
POPNET_INLINE vf32_t sqrt_f32(vf32_t a) { return std::sqrt(a); }

POPNET_INLINE vf32_t exp_f32(vf32_t a) { return std::exp(a); }
POPNET_INLINE vf32_t log_f32(vf32_t a) { return std::log(a); }

POPNET_INLINE vf32_t sin_f32(vf32_t a) { return std::sin(a); }
POPNET_INLINE vf32_t cos_f32(vf32_t a) { return std::cos(a); }
POPNET_INLINE vf32_t tanh_f32(vf32_t a) { return std::tanh(a); }

POPNET_INLINE vf32_t relu_f32(vf32_t a) { return std::max(a, 0.0f); }

POPNET_INLINE vf32_t sigmoid_f32(vf32_t a)
{
  const auto e = std::exp(-std::abs(a));
  return (a < 0 ? e : 1.0f) / (1.0f + e);
}

POPNET_INLINE vf32_t add_f32(vf32_t a, vf32_t b) { return a + b; }
POPNET_INLINE vf32_t sub_f32(vf32_t a, vf32_t b) { return a - b; }
POPNET_INLINE vf32_t mul_f32(vf32_t a, vf32_t b) { return a * b; }
POPNET_INLINE vf32_t div_f32(vf32_t a, vf32_t b) { return a / b; }
POPNET_INLINE vf32_t min_f32(vf32_t a, vf32_t b) { return std::min(a, b); }
POPNET_INLINE vf32_t max_f32(vf32_t a, vf32_t b) { return std::max(a, b); }

POPNET_INLINE vf32_t fma_f32(vf32_t a, vf32_t b, vf32_t c) { return std::fma(a, b, c); }

POPNET_INLINE f32_t reduce_sum_f32(vf32_t a) { return a; }
POPNET_INLINE f32_t reduce_product_f32(vf32_t a) { return a; }
POPNET_INLINE f32_t reduce_min_f32(vf32_t a) { return a; }
POPNET_INLINE f32_t reduce_max_f32(vf32_t a) { return a; }

#endif
} // namespace simd
