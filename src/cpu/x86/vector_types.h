#pragma once

#include "popnet.h"

#include <immintrin.h>
#include <intrin.h>
#include <cstdint>
#include <utility>

template<typename T>
struct hpair
{
  T fst;
  T snd;
};

struct vf32
{
  using register_t = __m512;
  using mask_t = __mmask16;

  static constexpr size_t elements = sizeof(register_t) / sizeof(float);

  POPNET_INLINE static vf32 load(const void* p) { return {_mm512_loadu_ps(p)}; }
  POPNET_INLINE static vf32 load(const void* p, mask_t m) { return {_mm512_maskz_loadu_ps(m, p)}; }
  POPNET_INLINE static hpair<vf32> load2(const void* pa, const void* pb) { return {vf32::load(pa), vf32::load(pb)}; }
  POPNET_INLINE static hpair<vf32> load2(const void* pa, const void* pb, mask_t m) { return {vf32::load(pa, m), vf32::load(pb, m)}; }

  POPNET_INLINE void store(void* p) const { _mm512_storeu_ps(p, v); }
  POPNET_INLINE void store(void* p, mask_t m) const { _mm512_mask_storeu_ps(p, m, v); }

  register_t v;
};

struct vi32
{
  using register_t = __m512i;
  using mask_t = __mmask16;

  static constexpr size_t elements = sizeof(register_t) / sizeof(int32_t);

  POPNET_INLINE static vi32 load(const void* p) { return {_mm512_loadu_epi32(p)}; }
  POPNET_INLINE static vi32 load(const void* p, mask_t m) { return {_mm512_maskz_loadu_epi32(m, p)}; }
  POPNET_INLINE static hpair<vi32> load2(const void* pa, const void* pb) { return {vi32::load(pa), vi32::load(pb)}; }
  POPNET_INLINE static hpair<vi32> load2(const void* pa, const void* pb, mask_t m) { return {vi32::load(pa, m), vi32::load(pb, m)}; }

  POPNET_INLINE void store(void* p) const { _mm512_storeu_epi32(p, v); }
  POPNET_INLINE void store(void* p, mask_t m) const { _mm512_mask_storeu_epi32(p, m, v); }

  register_t v;
};

struct vtrit 
{
  using register_t = __m512i;
  using mask_t = __mmask64;

  static constexpr size_t elements = sizeof(register_t) * 8;

  POPNET_INLINE static vtrit load(const void* p) 
  { 
    return {_mm512_loadu_si512(p), _mm512_loadu_si512(static_cast<const char*>(p) + 64)}; 
  }
  
  // TODO masked load (perhaps use byte interleaving here? The goal is to reduce memory alignment requirements, 1 byte is the smallest unit)

  POPNET_INLINE static hpair<vtrit> load2(const void* pa, const void* pb) 
  { 
    // Load both signs first for better memory access pattern
    const register_t as = _mm512_loadu_si512(pa);
    const register_t bs = _mm512_loadu_si512(pb);
    const register_t am = _mm512_loadu_si512(static_cast<const char*>(pa) + 64);
    const register_t bm = _mm512_loadu_si512(static_cast<const char*>(pb) + 64);
    return {vtrit{as, am}, vtrit{bs, bm}};
  }

  // TODO masked load2 (same comment as above)

  POPNET_INLINE void store(void* p) const { 
    _mm512_storeu_si512(p, sgn); 
    _mm512_storeu_si512(static_cast<char*>(p) + 64, mag); 
  }

  // TODO masked store

  register_t sgn; // sign bits
  register_t mag; // magnitude bits
};