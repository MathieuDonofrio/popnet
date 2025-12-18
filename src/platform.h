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