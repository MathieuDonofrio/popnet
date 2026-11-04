#pragma once

#include <cstdint>

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

#else
  static inline constexpr ::cpu_execution_model cpu_execution_model = ::cpu_execution_model::scalar;

  using vector_t = scalar_t;

  POPNET_INLINE static consteval size_t vector_length() { return 1; }

  POPNET_INLINE static vector_t vload(const void* p) { return load(p); }

  POPNET_INLINE static void vstore(void* p, vector_t v) { store(p, v); }

  POPNET_INLINE static scalar_t broadcast(scalar_t a) { return a; }
#endif

  POPNET_INLINE static consteval size_t element_size() { return sizeof(scalar_t); }

  POPNET_INLINE static scalar_t load(const void* p) { return *static_cast<const scalar_t*>(p); }

  POPNET_INLINE static void store(void* p, scalar_t v) { *static_cast<scalar_t*>(p) = v; }

  POPNET_INLINE static void* advance(void* p, size_t n) { return static_cast<scalar_t*>(p) + n; }
  POPNET_INLINE static const void* advance(const void* p, size_t n) { return static_cast<const scalar_t*>(p) + n; }
};

using f32_t = dtype_traits<popnet_dtype_f32>::scalar_t;
using vf32_t = dtype_traits<popnet_dtype_f32>::vector_t;
