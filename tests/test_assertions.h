#pragma once

#include <cmath>
#include <doctest/doctest.h>

// Finite, nonnegative tolerances are required. Evaluate in double to avoid F32
// overflow in the difference. Zero tolerances give numeric equality, including
// signed zeros; use bitwise comparisons when representation matters.
// NaNs never match; infinities match only when equal (including their sign).
inline bool f32_close(float actual, float expected, double atol, double rtol)
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
    const float popnet_check_f32_expected = (expected);                                                                \
    const double popnet_check_f32_atol = (atol);                                                                       \
    const double popnet_check_f32_rtol = (rtol);                                                                       \
    CHECK_MESSAGE(                                                                                                     \
        f32_close(popnet_check_f32_actual, popnet_check_f32_expected, popnet_check_f32_atol, popnet_check_f32_rtol),   \
        "actual=", popnet_check_f32_actual, ", expected=", popnet_check_f32_expected,                                  \
        ", atol=", popnet_check_f32_atol, ", rtol=", popnet_check_f32_rtol);                                           \
  }                                                                                                                    \
  while (false)
