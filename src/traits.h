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
template <popnet_type Type> struct traits;

template <> struct traits<POPNET_TYPE_F32>
{
  using scalar_t = float;

#if POPNET_AVX512
  static inline constexpr simd_category category = simd_category::fixed;

  using vector_t = __m512;
  using mask_t = __mmask16;

  POPNET_INLINE static consteval size_t vector_length() { return sizeof(vector_t) / element_size(); }

  POPNET_INLINE static vector_t loadv(const void* p) { return {_mm512_loadu_ps(p)}; }
  POPNET_INLINE static vector_t loadv(const void* p, const mask_t m) { return {_mm512_maskz_loadu_ps(m, p)}; }

  POPNET_INLINE static void storev(void* p, const vector_t v) { _mm512_storeu_ps(p, v); }
  POPNET_INLINE static void storev(void* p, const mask_t m, const vector_t v) { _mm512_mask_storeu_ps(p, m, v); }

  POPNET_INLINE static vector_t broadcast(scalar_t a) { return _mm512_set1_ps(a); }

  POPNET_INLINE static vector_t blend(const vector_t a, const vector_t b, const mask_t m)
  {
    return _mm512_mask_blend_ps(m, a, b);
  }

#else
  static inline constexpr simd_category category = simd_category::scalar;

  using vector_t = scalar_t;

  POPNET_INLINE static consteval size_t vector_length() { return 1; }
#endif

  POPNET_INLINE static consteval size_t element_size() { return sizeof(scalar_t); }

  POPNET_INLINE static scalar_t load(const void* p) { return *static_cast<const scalar_t*>(p); }

  POPNET_INLINE static void store(void* p, scalar_t v) { *static_cast<scalar_t*>(p) = v; }

  POPNET_INLINE static void* advance(void* p, size_t n) { return static_cast<scalar_t*>(p) + n; }
  POPNET_INLINE static const void* advance(const void* p, size_t n) { return static_cast<const scalar_t*>(p) + n; }
};

struct broadcast
{};

template <popnet_type dtype, typename T> auto maybe_broadcast_loadv(const T* p)
{
  using traits = traits<dtype>;

  if constexpr (std::same_as<T, broadcast>)
  {
    return traits::broadcast(traits::load(p));
  }
  else
  {
    return traits::loadv(p);
  }
}

template <popnet_type dtype, typename T> auto maybe_broadcast_loadv(const T* p, typename traits<dtype>::mask_t m)
{
  using traits = traits<dtype>;

  if constexpr (std::same_as<T, broadcast>)
  {
    return traits::broadcast(traits::load(p));
  }
  else
  {
    return traits::loadv(p, m);
  }
}

template <popnet_type dtype, typename T> const T* maybe_broadcast_advance(const T* p, size_t n)
{
  using traits = traits<dtype>;

  if constexpr (std::same_as<T, broadcast>)
  {
    return p;
  }
  else
  {
    return static_cast<const T*>(traits::advance(p, n));
  }
}

using f32_t = traits<POPNET_TYPE_F32>::scalar_t;
using vf32_t = traits<POPNET_TYPE_F32>::vector_t;

template <class T, class... Ts> constexpr T&& pack_first_element(T&& t, Ts&&...) noexcept { return std::forward<T>(t); }

template <auto op, popnet_type dtype, typename... T>
requires(sizeof...(T) >= 1)
POPNET_INLINE static void simd_map(void* POPNET_RESTRICT output, size_t n, const T* POPNET_RESTRICT... inputs) noexcept
{
  using traits = traits<dtype>;

  if constexpr (traits::category == simd_category::fixed)
  {
    constexpr size_t vector_length = traits::vector_length();

    const size_t tail = n % vector_length;
    const size_t body = n - tail;

    const void* const output_body_end = traits::advance(output, body);

    while (output != output_body_end)
    {
      const auto value = op(maybe_broadcast_loadv<dtype>(inputs)...);

      traits::storev(output, value);

      output = traits::advance(output, vector_length);
      ((inputs = maybe_broadcast_advance<dtype>(inputs, vector_length)), ...);
    }

    if (tail)
    {
      const traits::mask_t m = (1 << tail) - 1;

      const auto v = op(maybe_broadcast_loadv<dtype>(inputs, m)...);

      traits::storev(output, m, v);
    }
  }
  else if constexpr (traits::category == simd_category::scalar)
  {
    const void* const output_end = traits::advance(output, n);

    while (output != output_end)
    {
      const auto value = op(traits::load(inputs)...);

      traits::store(output, value);

      output = traits::advance(output, 1);
      ((inputs = maybe_broadcast_advance<dtype>(inputs, 1)), ...);
    }
  }
  else
  {
    static_assert([]() { return false; }, "SIMD category not supported yet");
  }
}

template <auto state_init_op, auto merge_op, auto horizontal_op, popnet_type dtype, typename... T>
requires(sizeof...(T) >= 1)
POPNET_INLINE static auto simd_reduce(size_t n, const T* POPNET_RESTRICT... inputs) noexcept
{
  using traits = traits<dtype>;

  auto s = state_init_op();

  if constexpr (traits::category == simd_category::fixed)
  {
    constexpr size_t vector_length = traits::vector_length();

    const size_t tail = n % vector_length;
    const size_t body = n - tail;

    const void* const first_input_end = traits::advance(pack_first_element(inputs...), body);

    while (pack_first_element(inputs...) != first_input_end)
    {
      s = merge_op(maybe_broadcast_loadv<dtype>(inputs)..., s);

      ((inputs = maybe_broadcast_advance<dtype>(inputs, vector_length)), ...);
    }

    if (tail)
    {
      const traits::mask_t m = (1 << tail) - 1;

      s = traits::blend(s, merge_op(maybe_broadcast_loadv<dtype>(inputs, m)..., s), m);
    }
  }
  else if constexpr (traits::category == simd_category::scalar)
  {
    const void* const first_input_end = traits::advance(pack_first_element(inputs...), n);

    while (pack_first_element(inputs...) != first_input_end)
    {
      s = merge_op(traits::load(inputs)..., s);

      ((inputs = maybe_broadcast_advance<dtype>(inputs, 1)), ...);
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
