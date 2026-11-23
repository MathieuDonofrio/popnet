#pragma once

#include "popnet.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(_M_ARM64) && !defined(__clang__)
#include <arm64_neon.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
#endif

// Compile-time target features, not runtime CPU detection.
#ifndef POPNET_SCALAR
#if defined(__AVX512F__) || defined(__AVX512__)
#define POPNET_AVX512 1
#elif defined(__AVX2__)
#define POPNET_AVX2 1
#elif defined(__AVX__)
#define POPNET_AVX 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define POPNET_SSE2 1
#elif defined(__ARM_FEATURE_SVE2)
#define POPNET_ARM_SVE2 1
#elif defined(__ARM_FEATURE_SVE)
#define POPNET_ARM_SVE 1
#elif (defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))) || \
    (defined(_M_ARM64) && !defined(__clang__))
#define POPNET_NEON 1
#else
#define POPNET_SCALAR 1
#endif
#endif
