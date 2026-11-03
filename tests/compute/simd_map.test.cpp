#include <doctest/doctest.h>
#include <vector>

#include "simd_algorithms.h"
#include "simd_ops_f32.h"

// @verifies REQ-0001: In-place unary traversal preserves the elementwise result.
// @verifies REQ-0002: In-place binary traversal preserves operand order and scalar broadcasting.
TEST_CASE("in-place elementwise traversal at SIMD boundaries")
{
  for (size_t n : {1, 3, 15, 16, 17, 31, 32, 33})
  {
    CAPTURE(n);

    // Guard both ends to detect stores beyond the requested range, including masked tails.
    std::vector<float> output(n + 2, -123.0f);
    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i)
    {
      output[i + 1] = static_cast<float>(i + 1);
      input[i] = static_cast<float>(2 * i + 3);
    }
    const float scalar = 3.0f;
    const auto* scalar_input = static_cast<const broadcast*>(static_cast<const void*>(&scalar));

    simd_map_inplace<simd::neg_f32, popnet_dtype_f32>(output.data() + 1, n);

    for (size_t i = 0; i < n; ++i)
      CHECK(output[i + 1] == -static_cast<float>(i + 1));
    CHECK(output.front() == -123.0f);
    CHECK(output.back() == -123.0f);

    for (size_t i = 0; i < n; ++i)
      output[i + 1] = static_cast<float>(i + 1);

    simd_map_inplace<simd::sub_f32, popnet_dtype_f32>(output.data() + 1, n, input.data());

    for (size_t i = 0; i < n; ++i)
      CHECK(output[i + 1] == -static_cast<float>(i + 2));
    CHECK(output.front() == -123.0f);
    CHECK(output.back() == -123.0f);

    for (size_t i = 0; i < n; ++i)
      output[i + 1] = static_cast<float>(i + 1);

    simd_map_inplace<simd::sub_f32, popnet_dtype_f32>(output.data() + 1, n, scalar_input);

    for (size_t i = 0; i < n; ++i)
      CHECK(output[i + 1] == static_cast<float>(i + 1) - scalar);
    CHECK(output.front() == -123.0f);
    CHECK(output.back() == -123.0f);
  }
}
