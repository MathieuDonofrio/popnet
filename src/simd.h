#include <algorithm>
#include <cmath>
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

template <typename T> struct hpair
{
  T fst;
  T snd;
};

template <popnet_type Type> struct simd;

using vt = simd<POPNET_TYPE_T>;
using vi8 = simd<POPNET_TYPE_I8>;
using vi16 = simd<POPNET_TYPE_I16>;
using vi32 = simd<POPNET_TYPE_I32>;
using vf16 = simd<POPNET_TYPE_F16>;
using vf32 = simd<POPNET_TYPE_F32>;

template <> struct simd<POPNET_TYPE_T>
{
  using scalar_t = int8_t;
#if POPNET_AVX512
  using reg_t = __m512i;
  using mask_t = __mmask64;
#else
  using reg_t = uintptr_t;
  using mask_t = uintptr_t;
#endif

  static constexpr size_t elements = sizeof(reg_t) * 8;
  static constexpr size_t size = sizeof(reg_t) * 2;

#if POPNET_AVX512
  POPNET_INLINE static vt load(const void* p)
  {
    return {_mm512_loadu_si512(p), _mm512_loadu_si512(static_cast<const char*>(p) + 64)};
  }

  // TODO masked load (perhaps use byte interleaving here? The goal is to reduce memory alignment requirements, 1 byte
  // is the smallest unit)

  POPNET_INLINE static hpair<vt> load2(const void* pa, const void* pb)
  {
    // Load both signs first for better memory access pattern
    const reg_t as = _mm512_loadu_si512(pa);
    const reg_t bs = _mm512_loadu_si512(pb);
    const reg_t am = _mm512_loadu_si512(static_cast<const char*>(pa) + 64);
    const reg_t bm = _mm512_loadu_si512(static_cast<const char*>(pb) + 64);
    return {vt{as, am}, vt { bs, bm }};
  }

  // TODO masked load2 (same comment as above)

  POPNET_INLINE void store(void* p) const
  {
    _mm512_storeu_si512(p, sgn);
    _mm512_storeu_si512(static_cast<char*>(p) + 64, mag);
  }

  // TODO masked store
#else

#endif

  reg_t sgn; // sign bits
  reg_t mag; // magnitude bits
};

template <> struct simd<POPNET_TYPE_F32>
{
  using scalar_t = float;
#if POPNET_AVX512
  using reg_t = __m512;
  using mask_t = __mmask16;
#else
  using reg_t = float;
  using mask_t = bool;
#endif

  static constexpr size_t elements = sizeof(reg_t) / sizeof(float);
  static constexpr size_t size = sizeof(reg_t);

#if POPNET_AVX512
  POPNET_INLINE static vf32 zero() { return {_mm512_setzero_ps()}; }

  POPNET_INLINE static vf32 load(const void* p) { return {_mm512_loadu_ps(p)}; }
  POPNET_INLINE static vf32 load(const void* p, mask_t m) { return {_mm512_maskz_loadu_ps(m, p)}; }
  POPNET_INLINE static hpair<vf32> load2(const void* pa, const void* pb) { return {vf32::load(pa), vf32::load(pb)}; }
  POPNET_INLINE static hpair<vf32> load2(const void* pa, const void* pb, mask_t m)
  {
    return {vf32::load(pa, m), vf32::load(pb, m)};
  }

  POPNET_INLINE void store(void* p) const { _mm512_storeu_ps(p, v); }
  POPNET_INLINE void store(void* p, mask_t m) const { _mm512_mask_storeu_ps(p, m, v); }

#else
  POPNET_INLINE static vf32 zero() { return {0}; }

  POPNET_INLINE static vf32 load(const void* p) { return {*reinterpret_cast<const reg_t*>(p)}; }
  POPNET_INLINE static vf32 load(const void* p, mask_t m) { return {*reinterpret_cast<const reg_t*>(p)}; }
  POPNET_INLINE static hpair<vf32> load2(const void* pa, const void* pb) { return {vf32::load(pa), vf32::load(pb)}; }
  POPNET_INLINE static hpair<vf32> load2(const void* pa, const void* pb, mask_t m)
  {
    return {vf32::load(pa, m), vf32::load(pb, m)};
  }

  POPNET_INLINE void store(void* p) const { *reinterpret_cast<reg_t*>(p) = v; }
  POPNET_INLINE void store(void* p, mask_t m) const { *reinterpret_cast<reg_t*>(p) = v; }
#endif

  reg_t v;
};

#if POPNET_AVX512

POPNET_INLINE vf32 vabs(vf32 a) { return vf32{_mm512_abs_ps(a.v)}; }
POPNET_INLINE vf32 vneg(vf32 a) { return vf32{_mm512_sub_ps(_mm512_setzero_ps(), a.v)}; }

POPNET_INLINE vf32 vsqr(vf32 a) { return vf32{_mm512_mul_ps(a.v, a.v)}; }
POPNET_INLINE vf32 vsqrt(vf32 a) { return vf32{_mm512_sqrt_ps(a.v)}; }

POPNET_INLINE vf32 vexp(vf32 a) { return vf32{_mm512_exp_ps(a.v)}; }
POPNET_INLINE vf32 vlog(vf32 a) { return vf32{_mm512_log_ps(a.v)}; }

POPNET_INLINE vf32 vsin(vf32 a) { return vf32{_mm512_sin_ps(a.v)}; }
POPNET_INLINE vf32 vcos(vf32 a) { return vf32{_mm512_cos_ps(a.v)}; }
POPNET_INLINE vf32 vtanh(vf32 a) { return vf32{_mm512_tanh_ps(a.v)}; }

POPNET_INLINE vf32 vrelu(vf32 a) { return vf32{_mm512_max_ps(a.v, _mm512_setzero_ps())}; }

POPNET_INLINE vf32 vsigmoid(vf32 a)
{
  const auto vneg = _mm512_sub_ps(_mm512_setzero_ps(), a.v);
  const auto vnegexp = _mm512_exp_ps(vneg);
  const auto vdenom = _mm512_add_ps(_mm512_set1_ps(1.0f), vnegexp);
  return vf32{_mm512_rcp14_ps(vdenom)}; // Approximation
}

POPNET_INLINE vf32 vadd(vf32 a, vf32 b) { return vf32{_mm512_add_ps(a.v, b.v)}; }
POPNET_INLINE vf32 vsub(vf32 a, vf32 b) { return vf32{_mm512_sub_ps(a.v, b.v)}; }
POPNET_INLINE vf32 vmul(vf32 a, vf32 b) { return vf32{_mm512_mul_ps(a.v, b.v)}; }
POPNET_INLINE vf32 vdiv(vf32 a, vf32 b) { return vf32{_mm512_div_ps(a.v, b.v)}; }

POPNET_INLINE vf32 vfma(vf32 a, vf32 b, vf32 c) { return vf32{_mm512_fmadd_ps(a.v, b.v, c.v)}; }

#else

POPNET_INLINE vf32 vabs(vf32 a) { return vf32{std::fabsf(a.v)}; }
POPNET_INLINE vf32 vneg(vf32 a) { return vf32{-a.v}; }

POPNET_INLINE vf32 vsqr(vf32 a) { return vf32{a.v * a.v}; }
POPNET_INLINE vf32 vsqrt(vf32 a) { return vf32{std::sqrtf(a.v)}; }

POPNET_INLINE vf32 vexp(vf32 a) { return vf32{std::expf(a.v)}; }
POPNET_INLINE vf32 vlog(vf32 a) { return vf32{std::logf(a.v)}; }

POPNET_INLINE vf32 vsin(vf32 a) { return vf32{std::sinf(a.v)}; }
POPNET_INLINE vf32 vcos(vf32 a) { return vf32{std::cosf(a.v)}; }
POPNET_INLINE vf32 vtanh(vf32 a) { return vf32{std::tanhf(a.v)}; }

POPNET_INLINE vf32 vrelu(vf32 a) { return vf32{std::max(a.v, 0.0f)}; }

POPNET_INLINE vf32 vsigmoid(vf32 a) { return vf32{1 / (1 + std::expf(-a.v))}; }

POPNET_INLINE vf32 vadd(vf32 a, vf32 b) { return vf32{a.v + b.v}; }
POPNET_INLINE vf32 vsub(vf32 a, vf32 b) { return vf32{a.v - b.v}; }
POPNET_INLINE vf32 vmul(vf32 a, vf32 b) { return vf32{a.v * b.v}; }
POPNET_INLINE vf32 vdiv(vf32 a, vf32 b) { return vf32{a.v / b.v}; }

POPNET_INLINE vf32 vfma(vf32 a, vf32 b, vf32 c) { return vf32{std::fmaf(a.v, b.v, c.v)}; }

#endif

template <auto Func, typename T, typename... TS>
POPNET_INLINE static void vtransform(T* POPNET_RESTRICT output, size_t n, const TS* POPNET_RESTRICT... inputs)
{
  const size_t full_vectors = n / T::elements;
  const size_t tail_elements = n % T::elements;

  for (size_t i = 0; i < full_vectors; i++)
  {
    auto b = Func(T::load(inputs + i)...);
    b.store(output + i);
  }

  if (tail_elements > 0)
  {
    const T::mask_t m = (1 << tail_elements) - 1;
    auto b = Func(T::load(inputs + full_vectors, m)...);
    b.store(output + full_vectors, m);
  }
}
