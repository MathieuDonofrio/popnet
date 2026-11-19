#pragma once

#include <cmath>
#include <cstddef>
#include <doctest/doctest.h>

// REQ-0026: approximate F32 mathematical functions.
inline constexpr double f32_math_atol = 1e-6;
inline constexpr double f32_math_rtol = 1e-5;

// The specification's gamma_j; pass binary64 unit roundoff for reference error.
inline double rounding_gamma(size_t count, double unit_roundoff = 0x1p-24)
{
  const double scaled = static_cast<double>(count) * unit_roundoff;
  return scaled / (1 - scaled);
}

// Finite, nonnegative tolerances are required. Evaluate in double to avoid F32
// overflow in the difference. Zero tolerances give numeric equality, including
// signed zeros; use bitwise comparisons when representation matters.
// NaNs never match; infinities match only when equal (including their sign).
// Keep higher-precision references instead of rounding them back to F32.
inline bool f32_close(float actual, double expected, double atol, double rtol)
{
  if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0)
  {
    return false;
  }
  if (!std::isfinite(actual) || !std::isfinite(expected))
  {
    return actual == expected;
  }

  return std::abs(double(actual) - double(expected)) <= atol + rtol * std::abs(double(expected));
}

// Capture arguments once; keep CHECK_MESSAGE here so doctest reports the call site.
#define CHECK_F32_CLOSE(actual, expected, atol, rtol)                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    const float popnet_check_f32_actual = (actual);                                                                    \
    const double popnet_check_f32_expected = (expected);                                                               \
    const double popnet_check_f32_atol = (atol);                                                                       \
    const double popnet_check_f32_rtol = (rtol);                                                                       \
    CHECK_MESSAGE(                                                                                                     \
        f32_close(popnet_check_f32_actual, popnet_check_f32_expected, popnet_check_f32_atol, popnet_check_f32_rtol),   \
        "actual=", popnet_check_f32_actual, ", expected=", popnet_check_f32_expected,                                  \
        ", atol=", popnet_check_f32_atol, ", rtol=", popnet_check_f32_rtol,                                            \
        ", error=", std::abs(double(popnet_check_f32_actual) - popnet_check_f32_expected),                             \
        ", bound=", popnet_check_f32_atol + popnet_check_f32_rtol * std::abs(popnet_check_f32_expected));              \
  }                                                                                                                    \
  while (false)
