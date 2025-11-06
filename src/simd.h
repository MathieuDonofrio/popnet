#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
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

  static constexpr size_t scalar_size = sizeof(scalar_t);

#if POPNET_AVX512
  static inline constexpr simd_category category = simd_category::fixed;

  using reg_t = __m512;
  using mask_t = __mmask16;

  static consteval size_t length() { return sizeof(reg_t) / scalar_size; }

  POPNET_INLINE static reg_t load(const void* p) { return {_mm512_loadu_ps(p)}; }
  POPNET_INLINE static reg_t load(const void* p, const mask_t m) { return {_mm512_maskz_loadu_ps(m, p)}; }

  POPNET_INLINE static void store(void* p, const reg_t v) { _mm512_storeu_ps(p, v); }
  POPNET_INLINE static void store(void* p, const mask_t m, const reg_t v) { _mm512_mask_storeu_ps(p, m, v); }

#else
  static inline constexpr simd_category category = simd_category::scalar;

  using reg_t = scalar_t;

  static consteval size_t length() { return 1; }
#endif
};

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
      const auto v = op(traits::load(reinterpret_cast<const char*>(inputs) + i * traits::scalar_size)...);
      traits::store(reinterpret_cast<char*>(output) + i * traits::scalar_size, v);
    }

    if (tail > 0)
    {
      const traits::mask_t m = (1 << tail) - 1;
      const auto v = op(traits::load(reinterpret_cast<const char*>(inputs) + body * traits::scalar_size, m)...);
      traits::store(reinterpret_cast<char*>(output) + body * traits::scalar_size, m, v);
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

using f32_t = simd_traits<POPNET_TYPE_F32>::scalar_t;
using vf32_t = simd_traits<POPNET_TYPE_F32>::reg_t;

#if POPNET_AVX512

POPNET_INLINE vf32_t vabs(vf32_t a) { return _mm512_abs_ps(a); }
POPNET_INLINE vf32_t vneg(vf32_t a) { return _mm512_sub_ps(_mm512_setzero_ps(), a); }

POPNET_INLINE vf32_t vsqr(vf32_t a) { return _mm512_mul_ps(a, a); }
POPNET_INLINE vf32_t vsqrt(vf32_t a) { return _mm512_sqrt_ps(a); }

POPNET_INLINE vf32_t vexp(vf32_t a) { return _mm512_exp_ps(a); }
POPNET_INLINE vf32_t vlog(vf32_t a) { return _mm512_log_ps(a); }

POPNET_INLINE vf32_t vsin(vf32_t a) { return _mm512_sin_ps(a); }
POPNET_INLINE vf32_t vcos(vf32_t a) { return _mm512_cos_ps(a); }
POPNET_INLINE vf32_t vtanh(vf32_t a) { return _mm512_tanh_ps(a); }

POPNET_INLINE vf32_t vrelu(vf32_t a) { return _mm512_max_ps(a, _mm512_setzero_ps()); }

POPNET_INLINE vf32_t vsigmoid(vf32_t a)
{
  const auto vneg = _mm512_sub_ps(_mm512_setzero_ps(), a);
  const auto vnegexp = _mm512_exp_ps(vneg);
  const auto vdenom = _mm512_add_ps(_mm512_set1_ps(1.0f), vnegexp);
  return _mm512_rcp14_ps(vdenom);
}

POPNET_INLINE vf32_t vadd(vf32_t a, vf32_t b) { return _mm512_add_ps(a, b); }
POPNET_INLINE vf32_t vsub(vf32_t a, vf32_t b) { return _mm512_sub_ps(a, b); }
POPNET_INLINE vf32_t vmul(vf32_t a, vf32_t b) { return _mm512_mul_ps(a, b); }
POPNET_INLINE vf32_t vdiv(vf32_t a, vf32_t b) { return _mm512_div_ps(a, b); }
POPNET_INLINE vf32_t vmin(vf32_t a, vf32_t b) { return _mm512_min_ps(a, b); }
POPNET_INLINE vf32_t vmax(vf32_t a, vf32_t b) { return _mm512_max_ps(a, b); }

POPNET_INLINE vf32_t vfma(vf32_t a, vf32_t b, vf32_t c) { return _mm512_fmadd_ps(a, b, c); }

POPNET_INLINE f32_t vreduce_sum(vf32_t a) { return _mm512_reduce_add_ps(a); }
POPNET_INLINE f32_t vreduce_min(vf32_t a) { return _mm512_reduce_min_ps(a); }
POPNET_INLINE f32_t vreduce_max(vf32_t a) { return _mm512_reduce_max_ps(a); }

#else

POPNET_INLINE vf32_t vabs(vf32_t a) { return std::fabsf(a); }
POPNET_INLINE vf32_t vneg(vf32_t a) { return -a; }

POPNET_INLINE vf32_t vsqr(vf32_t a) { return a * a; }
POPNET_INLINE vf32_t vsqrt(vf32_t a) { return std::sqrtf(a); }

POPNET_INLINE vf32_t vexp(vf32_t a) { return std::expf(a); }
POPNET_INLINE vf32_t vlog(vf32_t a) { return std::logf(a); }

POPNET_INLINE vf32_t vsin(vf32_t a) { return std::sinf(a); }
POPNET_INLINE vf32_t vcos(vf32_t a) { return std::cosf(a); }
POPNET_INLINE vf32_t vtanh(vf32_t a) { return std::tanhf(a); }

POPNET_INLINE vf32_t vrelu(vf32_t a) { return std::max(a, 0.0f); }

POPNET_INLINE vf32_t vsigmoid(vf32_t a) { return 1 / (1 + std::expf(-a)); }

POPNET_INLINE vf32_t vadd(vf32_t a, vf32_t b) { return a + b; }
POPNET_INLINE vf32_t vsub(vf32_t a, vf32_t b) { return a - b; }
POPNET_INLINE vf32_t vmul(vf32_t a, vf32_t b) { return a * b; }
POPNET_INLINE vf32_t vdiv(vf32_t a, vf32_t b) { return a / b; }
POPNET_INLINE vf32_t vmin(vf32_t a, vf32_t b) { return std::min(a, b); }
POPNET_INLINE vf32_t vmax(vf32_t a, vf32_t b) { return std::max(a, b); }

POPNET_INLINE vf32_t vfma(vf32_t a, vf32_t b, vf32_t c) { return std::fmaf(a, b, c); }

POPNET_INLINE f32_t vreduce_sum(vf32_t a) { return a; }
POPNET_INLINE f32_t vreduce_min(vf32_t a) { return a; }
POPNET_INLINE f32_t vreduce_max(vf32_t a) { return a; }

#endif
