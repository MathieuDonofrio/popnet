#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <utility>

#include "platform.h"

enum class cpu_execution_model
{
  scalar,
  fixed,
  variable
};

template <popnet_dtype Type> struct dtype_traits;

template <> struct dtype_traits<popnet_dtype_f32>
{
  using scalar_t = float;

#if POPNET_AVX512
  static inline constexpr cpu_execution_model cpu_execution_model = cpu_execution_model::fixed;

  using vector_t = __m512;
  using mask_t = __mmask16;

  POPNET_INLINE static consteval size_t vector_length() { return sizeof(vector_t) / element_size(); }

  POPNET_INLINE static vector_t vload(const void* p) { return {_mm512_loadu_ps(p)}; }
  POPNET_INLINE static vector_t vload(const void* p, const mask_t m) { return {_mm512_maskz_loadu_ps(m, p)}; }

  POPNET_INLINE static void vstore(void* p, const vector_t v) { _mm512_storeu_ps(p, v); }
  POPNET_INLINE static void vstore(void* p, const mask_t m, const vector_t v) { _mm512_mask_storeu_ps(p, m, v); }

  POPNET_INLINE static vector_t broadcast(scalar_t a) { return _mm512_set1_ps(a); }

  POPNET_INLINE static vector_t blend(const vector_t a, const vector_t b, const mask_t m)
  {
    return _mm512_mask_blend_ps(m, a, b);
  }

#else
  static inline constexpr cpu_execution_model cpu_execution_model = cpu_execution_model::scalar;

  using vector_t = scalar_t;

  POPNET_INLINE static consteval size_t vector_length() { return 1; }
#endif

  POPNET_INLINE static consteval size_t element_size() { return sizeof(scalar_t); }

  POPNET_INLINE static scalar_t load(const void* p) { return *static_cast<const scalar_t*>(p); }

  POPNET_INLINE static void store(void* p, scalar_t v) { *static_cast<scalar_t*>(p) = v; }

  POPNET_INLINE static void* advance(void* p, size_t n) { return static_cast<scalar_t*>(p) + n; }
  POPNET_INLINE static const void* advance(const void* p, size_t n) { return static_cast<const scalar_t*>(p) + n; }
};

using f32_t = dtype_traits<popnet_dtype_f32>::scalar_t;
using vf32_t = dtype_traits<popnet_dtype_f32>::vector_t;

#if POPNET_AVX512

POPNET_INLINE vf32_t vset_zero_f32() { return _mm512_setzero_ps(); }
POPNET_INLINE vf32_t vset_one_f32() { return _mm512_set1_ps(1.0f); }
POPNET_INLINE vf32_t vset_neg_inf_f32() { return _mm512_set1_ps(-std::numeric_limits<f32_t>::infinity()); }
POPNET_INLINE vf32_t vset_pos_inf_f32() { return _mm512_set1_ps(std::numeric_limits<f32_t>::infinity()); }

POPNET_INLINE vf32_t vabs_f32(vf32_t a) { return _mm512_abs_ps(a); }
POPNET_INLINE vf32_t vneg_f32(vf32_t a) { return _mm512_sub_ps(_mm512_setzero_ps(), a); }

POPNET_INLINE vf32_t vsqr_f32(vf32_t a) { return _mm512_mul_ps(a, a); }
POPNET_INLINE vf32_t vsqrt_f32(vf32_t a) { return _mm512_sqrt_ps(a); }

POPNET_INLINE vf32_t vexp_f32(vf32_t a) { return _mm512_exp_ps(a); }
POPNET_INLINE vf32_t vlog_f32(vf32_t a) { return _mm512_log_ps(a); }

POPNET_INLINE vf32_t vsin_f32(vf32_t a) { return _mm512_sin_ps(a); }
POPNET_INLINE vf32_t vcos_f32(vf32_t a) { return _mm512_cos_ps(a); }
POPNET_INLINE vf32_t vtanh_f32(vf32_t a) { return _mm512_tanh_ps(a); }

POPNET_INLINE vf32_t vrelu_f32(vf32_t a) { return _mm512_max_ps(a, _mm512_setzero_ps()); }

POPNET_INLINE vf32_t vsigmoid_f32(vf32_t a)
{
  const auto vneg = _mm512_sub_ps(_mm512_setzero_ps(), a);
  const auto vnegexp = _mm512_exp_ps(vneg);
  const auto vdenom = _mm512_add_ps(_mm512_set1_ps(1.0f), vnegexp);
  return _mm512_rcp14_ps(vdenom);
}

POPNET_INLINE vf32_t vadd_f32(vf32_t a, vf32_t b) { return _mm512_add_ps(a, b); }
POPNET_INLINE vf32_t vsub_f32(vf32_t a, vf32_t b) { return _mm512_sub_ps(a, b); }
POPNET_INLINE vf32_t vmul_f32(vf32_t a, vf32_t b) { return _mm512_mul_ps(a, b); }
POPNET_INLINE vf32_t vdiv_f32(vf32_t a, vf32_t b) { return _mm512_div_ps(a, b); }
POPNET_INLINE vf32_t vmin_f32(vf32_t a, vf32_t b) { return _mm512_min_ps(a, b); }
POPNET_INLINE vf32_t vmax_f32(vf32_t a, vf32_t b) { return _mm512_max_ps(a, b); }

POPNET_INLINE vf32_t vfma_f32(vf32_t a, vf32_t b, vf32_t c) { return _mm512_fmadd_ps(a, b, c); }

POPNET_INLINE f32_t vreduce_sum_f32(vf32_t a) { return _mm512_reduce_add_ps(a); }
POPNET_INLINE f32_t vreduce_mean_f32(vf32_t a, size_t n) { return _mm512_reduce_add_ps(a) / n; }
POPNET_INLINE f32_t vreduce_product_f32(vf32_t a) { return _mm512_reduce_mul_ps(a); }
POPNET_INLINE f32_t vreduce_min_f32(vf32_t a) { return _mm512_reduce_min_ps(a); }
POPNET_INLINE f32_t vreduce_max_f32(vf32_t a) { return _mm512_reduce_max_ps(a); }

#else

POPNET_INLINE vf32_t vset_zero_f32() { return 0; }
POPNET_INLINE vf32_t vset_one_f32() { return 1; }
POPNET_INLINE vf32_t vset_neg_inf_f32() { return -std::numeric_limits<f32_t>::infinity(); }
POPNET_INLINE vf32_t vset_pos_inf_f32() { return std::numeric_limits<f32_t>::infinity(); }

POPNET_INLINE vf32_t vabs_f32(vf32_t a) { return std::fabsf(a); }
POPNET_INLINE vf32_t vneg_f32(vf32_t a) { return -a; }

POPNET_INLINE vf32_t vsqr_f32(vf32_t a) { return a * a; }
POPNET_INLINE vf32_t vsqrt_f32(vf32_t a) { return std::sqrtf(a); }

POPNET_INLINE vf32_t vexp_f32(vf32_t a) { return std::expf(a); }
POPNET_INLINE vf32_t vlog_f32(vf32_t a) { return std::logf(a); }

POPNET_INLINE vf32_t vsin_f32(vf32_t a) { return std::sinf(a); }
POPNET_INLINE vf32_t vcos_f32(vf32_t a) { return std::cosf(a); }
POPNET_INLINE vf32_t vtanh_f32(vf32_t a) { return std::tanhf(a); }

POPNET_INLINE vf32_t vrelu_f32(vf32_t a) { return std::max(a, 0.0f); }

POPNET_INLINE vf32_t vsigmoid_f32(vf32_t a) { return 1 / (1 + std::expf(-a)); }

POPNET_INLINE vf32_t vadd_f32(vf32_t a, vf32_t b) { return a + b; }
POPNET_INLINE vf32_t vsub_f32(vf32_t a, vf32_t b) { return a - b; }
POPNET_INLINE vf32_t vmul_f32(vf32_t a, vf32_t b) { return a * b; }
POPNET_INLINE vf32_t vdiv_f32(vf32_t a, vf32_t b) { return a / b; }
POPNET_INLINE vf32_t vmin_f32(vf32_t a, vf32_t b) { return std::min(a, b); }
POPNET_INLINE vf32_t vmax_f32(vf32_t a, vf32_t b) { return std::max(a, b); }

POPNET_INLINE vf32_t vfma_f32(vf32_t a, vf32_t b, vf32_t c) { return std::fmaf(a, b, c); }

POPNET_INLINE f32_t vreduce_sum_f32(vf32_t a) { return a; }
POPNET_INLINE f32_t vreduce_mean_f32(vf32_t a, size_t n) { return a / n; }
POPNET_INLINE f32_t vreduce_product_f32(vf32_t a) { return a; }
POPNET_INLINE f32_t vreduce_min_f32(vf32_t a) { return a; }
POPNET_INLINE f32_t vreduce_max_f32(vf32_t a) { return a; }

#endif
