#include "simd_algorithms.h"

#include <doctest/doctest.h>
#include <vector>

// Internal algorithm coverage: no standalone requirement specifies this helper's interface.
TEST_CASE("in-place SIMD maps handle broadcast operands and tails")
{
  for (int operation = 0; operation < 4; ++operation)
    for (const size_t n : {0, 1, 15, 16, 17, 31, 32, 33})
    {
      CAPTURE(n);
      CAPTURE(operation);
      std::vector<float> storage(n + 2, -123.0f);
      float* output = storage.data() + 1;
      std::vector<float> input(n);
      for (size_t i = 0; i < n; ++i)
      {
        output[i] = static_cast<float>(i + 1);
        input[i] = static_cast<float>(i + 2);
      }
      const auto original = storage;
      const auto original_input = input;
      const float divisor = 3.0f;
      const auto* scalar = static_cast<const broadcast*>(static_cast<const void*>(&divisor));
      std::vector<float> expected(n);

      if (operation == 0)
      {
        for (size_t i = 0; i < n; ++i)
          expected[i] = original[i + 1] * original[i + 1];

        simd_map_inplace<simd::sqr_f32, popnet_dtype_f32>(output, n);
      }
      else if (operation == 1)
      {
        for (size_t i = 0; i < n; ++i)
          expected[i] = original[i + 1] + input[i];

        simd_map_inplace<simd::add_f32, popnet_dtype_f32>(output, n, input.data());
      }
      else if (operation == 2)
      {
        for (size_t i = 0; i < n; ++i)
          expected[i] = original[i + 1] / divisor;

        simd_map_inplace<simd::div_f32, popnet_dtype_f32>(output, n, scalar);
      }
      else
      {
        for (size_t i = 0; i < n; ++i)
          expected[i] = original[i + 1] * input[i] + divisor;

        simd_map_inplace<simd::fma_f32, popnet_dtype_f32>(output, n, input.data(), scalar);
      }

      for (size_t i = 0; i < n; ++i)
        CHECK(output[i] == doctest::Approx(expected[i]));
      CHECK(storage.front() == original.front());
      CHECK(storage.back() == original.back());
      CHECK(input == original_input);
      CHECK(divisor == 3.0f);
    }
}
