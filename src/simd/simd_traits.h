#pragma once

#include <cstdint>

#include "platform.h"

enum class cpu_execution_model
{
  scalar,
  fixed,
  variable
};

template <popnet_dtype Type> struct simd_traits;

template <> struct simd_traits<popnet_dtype_f32>
{
  using scalar_t = float;

  POPNET_INLINE static consteval size_t element_size() { return sizeof(scalar_t); }

#if POPNET_AVX512
  static inline constexpr ::cpu_execution_model cpu_execution_model = ::cpu_execution_model::fixed;

  using vector_t = __m512;
  using mask_t = __mmask16;

  POPNET_INLINE static consteval size_t vector_length() { return sizeof(vector_t) / element_size(); }

  // Enable the first count lanes, where 0 <= count <= vector_length() (16).
  POPNET_INLINE static mask_t tail_mask(size_t count) { return static_cast<mask_t>((uint32_t{1} << count) - 1); }

  POPNET_INLINE static vector_t vload(const void* p) { return _mm512_loadu_ps(p); }
  POPNET_INLINE static vector_t vload(const void* p, const mask_t m) { return _mm512_maskz_loadu_ps(m, p); }

  POPNET_INLINE static void vstore(void* p, const vector_t v) { _mm512_storeu_ps(p, v); }
  POPNET_INLINE static void vstore(void* p, const mask_t m, const vector_t v) { _mm512_mask_storeu_ps(p, m, v); }

  POPNET_INLINE static vector_t broadcast(scalar_t a) { return _mm512_set1_ps(a); }

  POPNET_INLINE static vector_t blend(const vector_t a, const vector_t b, const mask_t m)
  {
    return _mm512_mask_blend_ps(m, a, b);
  }

#elif POPNET_AVX2 || POPNET_AVX
  static inline constexpr ::cpu_execution_model cpu_execution_model = ::cpu_execution_model::fixed;

  using vector_t = __m256;
  using mask_t = __m256i;

  POPNET_INLINE static consteval size_t vector_length() { return sizeof(vector_t) / element_size(); }

  POPNET_INLINE static mask_t tail_mask(size_t count)
  {
    const auto lanes = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    return _mm256_castps_si256(_mm256_cmp_ps(lanes, _mm256_set1_ps(static_cast<float>(count)), _CMP_LT_OQ));
  }

  POPNET_INLINE static vector_t vload(const void* p) { return _mm256_loadu_ps(static_cast<const scalar_t*>(p)); }
  POPNET_INLINE static vector_t vload(const void* p, mask_t m)
  {
    return _mm256_maskload_ps(static_cast<const scalar_t*>(p), m);
  }

  POPNET_INLINE static void vstore(void* p, vector_t v) { _mm256_storeu_ps(static_cast<scalar_t*>(p), v); }
  POPNET_INLINE static void vstore(void* p, mask_t m, vector_t v)
  {
    _mm256_maskstore_ps(static_cast<scalar_t*>(p), m, v);
  }

  POPNET_INLINE static vector_t broadcast(scalar_t a) { return _mm256_set1_ps(a); }

  POPNET_INLINE static vector_t blend(vector_t a, vector_t b, mask_t m)
  {
    return _mm256_blendv_ps(a, b, _mm256_castsi256_ps(m));
  }

#elif POPNET_ARM_SVE || POPNET_ARM_SVE2
  static inline constexpr ::cpu_execution_model cpu_execution_model = ::cpu_execution_model::variable;

  using vector_t = svfloat32_t;
  using mask_t = svbool_t;

  POPNET_INLINE static size_t vector_length() { return svcntw(); }

  POPNET_INLINE static mask_t tail_mask(size_t count) { return svwhilelt_b32_u64(0, count); }

  POPNET_INLINE static vector_t vload(const void* p, mask_t m)
  {
    return svld1_f32(m, static_cast<const scalar_t*>(p));
  }
  POPNET_INLINE static vector_t vload(const void* p) { return vload(p, svptrue_b32()); }

  POPNET_INLINE static void vstore(void* p, mask_t m, vector_t v) { svst1_f32(m, static_cast<scalar_t*>(p), v); }
  POPNET_INLINE static void vstore(void* p, vector_t v) { vstore(p, svptrue_b32(), v); }

  POPNET_INLINE static vector_t broadcast(scalar_t a) { return svdup_n_f32(a); }

  POPNET_INLINE static vector_t blend(vector_t a, vector_t b, mask_t m) { return svsel_f32(m, b, a); }

#elif POPNET_SSE2 || POPNET_NEON
  static inline constexpr ::cpu_execution_model cpu_execution_model = ::cpu_execution_model::fixed;

#if POPNET_NEON
  using vector_t = float32x4_t;

  POPNET_INLINE static vector_t vload(const void* p) { return vld1q_f32(static_cast<const scalar_t*>(p)); }
  POPNET_INLINE static void vstore(void* p, vector_t v) { vst1q_f32(static_cast<scalar_t*>(p), v); }

  POPNET_INLINE static vector_t broadcast(scalar_t a) { return vdupq_n_f32(a); }
#else
  using vector_t = __m128;

  POPNET_INLINE static vector_t vload(const void* p) { return _mm_loadu_ps(static_cast<const scalar_t*>(p)); }
  POPNET_INLINE static void vstore(void* p, vector_t v) { _mm_storeu_ps(static_cast<scalar_t*>(p), v); }

  POPNET_INLINE static vector_t broadcast(scalar_t a) { return _mm_set1_ps(a); }
#endif
  using mask_t = size_t;

  POPNET_INLINE static consteval size_t vector_length() { return sizeof(vector_t) / element_size(); }

  POPNET_INLINE static mask_t tail_mask(size_t count) { return count; }

  // Stage only the active lanes when masked memory operations are unavailable.
  POPNET_INLINE static void copy_tail(scalar_t* dst, const scalar_t* src, mask_t m)
  {
    for (size_t lane = 0; lane < vector_length(); ++lane)
    {
      if (lane < m) dst[lane] = src[lane];
    }
  }

  POPNET_INLINE static vector_t vload(const void* p, mask_t m)
  {
    scalar_t lanes[4]{};
    copy_tail(lanes, static_cast<const scalar_t*>(p), m);
    return vload(lanes);
  }

  POPNET_INLINE static void vstore(void* p, mask_t m, vector_t v)
  {
    scalar_t lanes[4];
    vstore(lanes, v);
    copy_tail(static_cast<scalar_t*>(p), lanes, m);
  }

  POPNET_INLINE static vector_t blend(vector_t a, vector_t b, mask_t m)
  {
#if POPNET_NEON
    const uint32_t lanes[] = {0, 1, 2, 3};
    const auto active = vcgtq_u32(vdupq_n_u32(static_cast<uint32_t>(m)), vld1q_u32(lanes));
    return vbslq_f32(active, b, a);
#else
    const auto active = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_set1_epi32(static_cast<int>(m)), _mm_setr_epi32(0, 1, 2, 3)));
    return _mm_or_ps(_mm_andnot_ps(active, a), _mm_and_ps(active, b));
#endif
  }

#else
  static inline constexpr ::cpu_execution_model cpu_execution_model = ::cpu_execution_model::scalar;

  using vector_t = scalar_t;

  POPNET_INLINE static consteval size_t vector_length() { return 1; }

  POPNET_INLINE static vector_t vload(const void* p) { return load(p); }

  POPNET_INLINE static void vstore(void* p, vector_t v) { store(p, v); }

  POPNET_INLINE static scalar_t broadcast(scalar_t a) { return a; }
#endif

  POPNET_INLINE static scalar_t load(const void* p) { return *static_cast<const scalar_t*>(p); }

  POPNET_INLINE static void store(void* p, scalar_t v) { *static_cast<scalar_t*>(p) = v; }

  POPNET_INLINE static void* advance(void* p, size_t n) { return static_cast<scalar_t*>(p) + n; }
  POPNET_INLINE static const void* advance(const void* p, size_t n) { return static_cast<const scalar_t*>(p) + n; }
};

using f32_t = simd_traits<popnet_dtype_f32>::scalar_t;
using vf32_t = simd_traits<popnet_dtype_f32>::vector_t;
