#pragma once

#include <benchmark/benchmark.h>
#include <vector>

#include "tensor.h"

// Bounded, positive, nonconstant values also keep sqrt inputs in its domain.
inline std::vector<float> bench_input(size_t n)
{
  std::vector<float> values(n);
  for (size_t i = 0; i < n; ++i)
    values[i] = 0.25f + static_cast<float>((i * 17) % 101) / 101.0f;
  return values;
}
