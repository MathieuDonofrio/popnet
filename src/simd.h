#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <utility>

#include "popnet.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
#endif

#if defined(__riscv_vector) || defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

#if defined(__AVX512F__) || defined(__AVX512__)
#define POPNET_AVX512 1
#elif defined(__AVX2__)
#define POPNET_AVX2 1
#elif defined(__AVX__)
#define POPNET_AVX 1
#elif defined(__SSE2__)
#define POPNET_SSE2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define POPNET_NEON 1
#elif defined(__ARM_FEATURE_SVE)
#define POPNET_ARM_SVE 1
#elif defined(__riscv_vector) || defined(__riscv_v_intrinsic)
#define POPNET_RISCV_SVE 1
#else
#define POPNET_SCALAR 1
#endif

enum class simd_category
{
  scalar,
  fixed,
  variable
};

template <popnet_type Type> struct simd_traits;

template <> struct simd_traits<POPNET_TYPE_F32>
{
  using scalar_t = float;

#if POPNET_AVX512
  static inline constexpr simd_category category = simd_category::fixed;

  using vector_t = __m512;
  using mask_t = __mmask16;

  POPNET_INLINE static consteval size_t element_size() { return sizeof(scalar_t); }
  POPNET_INLINE static consteval size_t length() { return sizeof(vector_t) / element_size(); }

  POPNET_INLINE static vector_t load(const void* p) { return {_mm512_loadu_ps(p)}; }
  POPNET_INLINE static vector_t load(const void* p, const mask_t m) { return {_mm512_maskz_loadu_ps(m, p)}; }

  POPNET_INLINE static void store(void* p, const vector_t v) { _mm512_storeu_ps(p, v); }
  POPNET_INLINE static void store(void* p, const mask_t m, const vector_t v) { _mm512_mask_storeu_ps(p, m, v); }

#else
  static inline constexpr simd_category category = simd_category::scalar;

  using vector_t = scalar_t;

  POPNET_INLINE static consteval size_t length() { return 1; }
#endif
};

using f32_t = simd_traits<POPNET_TYPE_F32>::scalar_t;
using vf32_t = simd_traits<POPNET_TYPE_F32>::vector_t;

template <auto op, popnet_type type, std::same_as<void>... T>
POPNET_INLINE static void simd_map(void* POPNET_RESTRICT output, size_t n, const T* POPNET_RESTRICT... inputs)
{
  using traits = simd_traits<type>;

  if constexpr (traits::category == simd_category::fixed)
  {
    const size_t tail = n % traits::length();
    const size_t body = n - tail;

    for (size_t i = 0; i < body; i += traits::length())
    {
      const auto v = op(traits::load(reinterpret_cast<const char*>(inputs) + i * traits::element_size())...);
      traits::store(reinterpret_cast<char*>(output) + i * traits::element_size(), v);
    }

    if (tail)
    {
      const traits::mask_t m = (1 << tail) - 1;
      const auto v = op(traits::load(reinterpret_cast<const char*>(inputs) + body * traits::element_size(), m)...);
      traits::store(reinterpret_cast<char*>(output) + body * traits::element_size(), m, v);
    }
  }
  else if constexpr (traits::category == simd_category::scalar)
  {
    for (size_t i = 0; i < n; ++i)
    {
      const auto v = op(reinterpret_cast<const traits::scalar_t*>(inputs)[i]...);
      reinterpret_cast<traits::scalar_t*>(output)[i] = v;
    }
  }
  else
  {
    static_assert([]() { return false; }, "SIMD category not supported yet");
  }
}

template <auto state_init_op, auto merge_op, auto horizontal_op, popnet_type type, std::same_as<void>... T>
POPNET_INLINE static auto simd_reduce(size_t n, const T* POPNET_RESTRICT... inputs)
{
  using traits = simd_traits<type>;

  auto s = state_init_op();

  if constexpr (traits::category == simd_category::fixed)
  {
    const size_t tail = n % traits::length();
    const size_t body = n - tail;

    for (size_t i = 0; i < body; i += traits::length())
    {
      s = merge_op(s, traits::load(reinterpret_cast<const char*>(inputs) + i * traits::element_size())...);
    }

    if (tail)
    {
      const traits::mask_t m = (1 << tail) - 1;
      s = merge_op(s, traits::load(reinterpret_cast<const char*>(inputs) + body * traits::element_size(), m)...);
    }
  }
  else if constexpr (traits::category == simd_category::scalar)
  {
    for (size_t i = 0; i < n; ++i)
    {
      s = merge_op(s, reinterpret_cast<const traits::scalar_t*>(inputs)[i]...);
    }
  }
  else
  {
    static_assert([]() { return false; }, "SIMD category not supported yet");
  }

  if constexpr (std::is_invocable_v<decltype(horizontal_op), decltype(s), size_t>)
  {
    return horizontal_op(s, n);
  }
  else
  {
    return horizontal_op(s);
  }
}

#if POPNET_AVX512

POPNET_INLINE vf32_t vset_f32(f32_t a) { return _mm512_set1_ps(a); }
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
POPNET_INLINE f32_t vreduce_mean_f32(vf32_t a, f32_t n) { return _mm512_reduce_add_ps(a) / n; }
POPNET_INLINE f32_t vreduce_product_f32(vf32_t a) { return _mm512_reduce_mul_ps(a); }
POPNET_INLINE f32_t vreduce_min_f32(vf32_t a) { return _mm512_reduce_min_ps(a); }
POPNET_INLINE f32_t vreduce_max_f32(vf32_t a) { return _mm512_reduce_max_ps(a); }

#else

POPNET_INLINE vf32_t vset_f32(f32_t a) { return a; }
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
POPNET_INLINE f32_t vreduce_mean_f32(vf32_t a, f32_t n) { return a / n; }
POPNET_INLINE f32_t vreduce_product_f32(vf32_t a) { return a; }
POPNET_INLINE f32_t vreduce_min_f32(vf32_t a) { return a; }
POPNET_INLINE f32_t vreduce_max_f32(vf32_t a) { return a; }

#endif
