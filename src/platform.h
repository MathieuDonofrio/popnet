#pragma once

#if defined(POPNET_DISABLE_SIMD)
#define POPNET_SCALAR 1
#elif defined(__AVX512F__) && !defined(__AVX10_VER__)
// MSVC also advertises AVX-512F for AVX10 targets that may lack 512-bit vectors.
#include <immintrin.h>
#define POPNET_AVX512 1
#elif defined(__AVX__)
#include <immintrin.h>
#define POPNET_AVX 1
#elif defined(__SSE2__) || (defined(_MSC_VER) && !defined(__clang__) && \
                           ((defined(_M_X64) && !defined(_M_ARM64EC)) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)))
#include <emmintrin.h>
#define POPNET_SSE2 1
#elif defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define POPNET_SVE 1
#elif defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define POPNET_NEON 1
#elif defined(_M_ARM64) && !defined(__clang__)
#include <arm64_neon.h>
#define POPNET_NEON 1
#else
#define POPNET_SCALAR 1
#endif
