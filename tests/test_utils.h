#pragma once

#include <algorithm>
#include <numbers>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

#include "dtype_traits.h"
#include "popnet.h"

template <popnet_dtype dtype>
static auto generate_scalar_data(size_t n, auto lower, auto upper, bool exclude_zero, size_t index = 0)
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = typename dtype_traits::scalar_t;

  const size_t seed = 0x9E3779B9u * (index + 1);
  const size_t zero_stride = 3 + (index % 7);
  const bool contains_zeros = !exclude_zero && lower <= 0 && upper >= 0;

  std::mt19937 rng(seed);

  std::vector<scalar_t> data;
  data.reserve(n);

  switch (dtype)
  {
  case popnet_dtype_f32:
  {
    std::uniform_real_distribution<float> dist(lower, upper);

    for (size_t i = 0; i < n; ++i)
    {
      scalar_t value;

      if (contains_zeros && i % zero_stride == 0)
      {
        value = 0.0f;
      }
      else
      {
        do
        {
          value = dist(rng);
        }
        while (value == 0.0f);
      }

      data.push_back(value);
    }

    break;
  }
  default: FAIL_CHECK("Unsupported dtype in generate_scalar_data"); break;
  }

  return data;
}
