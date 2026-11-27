#include <algorithm>
#include <cmath>
#include <doctest/doctest.h>
#include <vector>

#include "popnet.h"

namespace
{
constexpr size_t widths[] = {1,  2,   3,   4,   5,   7,   8,   9,   12,   63,   64,
                             65, 255, 256, 300, 511, 512, 513, 900, 1023, 1024, 1025};
} // namespace

// @verifies REQ-0015: An offset write changes only the selected i8 input element.
// @verifies REQ-0016: Reading i8 data at a logical offset returns the requested values across rows.
// @verifies REQ-0017: Cast construction records the cast operation and sets result metadata.
// @verifies REQ-0020: Forward execution evaluates chained casts and uses updated inputs on re-execution.
// @verifies REQ-0024: Cast results preserve the input shape and use the requested destination dtype.
// @verifies REQ-0026: Ternary casts preserve allowed values and canonicalize zero, and f32 to i8 truncates toward zero.
TEST_CASE("tensor casts preserve shape and convert values")
{
  SUBCASE("f32 to ternary and back")
  {
    for (size_t width : widths)
    {
      CAPTURE(width);
      const popnet_shape shape{{2, 2, 1, width}};
      const size_t count = popnet_numel(shape);
      auto* input = popnet_empty(popnet_dtype_f32, shape);
      REQUIRE(input != nullptr);
      auto* packed = popnet_cast(input, popnet_dtype_ternary);
      REQUIRE(packed != nullptr);
      auto* output = popnet_cast(packed, popnet_dtype_f32);
      REQUIRE(output != nullptr);
      CHECK(popnet_opof(packed) == popnet_op_cast);
      CHECK(std::ranges::equal(popnet_shapeof(packed).dim, shape.dim));
      CHECK(popnet_dtypeof(packed) == popnet_dtype_ternary);
      CHECK(std::ranges::equal(popnet_shapeof(output).dim, shape.dim));
      CHECK(popnet_dtypeof(output) == popnet_dtype_f32);
      auto* graph = popnet_graph_alloc();
      REQUIRE(graph != nullptr);
      REQUIRE(popnet_graph_build_forward(graph, output));

      std::vector<float> values(count);
      for (size_t i = 0; i < count; ++i)
        values[i] = static_cast<float>(static_cast<int>(i % 3) - 1);
      popnet_set_data(input, values.data(), 0, count);
      popnet_graph_compute_forward(graph);

      std::vector<float> result(count);
      popnet_get_data(output, result.data(), 0, count);
      CHECK(result == values);

      std::fill(values.begin(), values.end(), -0.0f);
      popnet_set_data(input, values.data(), 0, count);
      popnet_graph_compute_forward(graph);
      popnet_get_data(output, result.data(), 0, count);

      CHECK(std::all_of(result.begin(), result.end(), [](float v) { return v == 0 && !std::signbit(v); }));

      popnet_graph_free(graph);
      popnet_free(output);
      popnet_free(packed);
      popnet_free(input);
    }
  }

  SUBCASE("i8 to ternary and back")
  {
    const popnet_shape shape{{1, 1, 2, 513}};
    std::vector<int8_t> values(1026);
    values[511] = -1;
    values[512] = 1;
    values[513] = -1;
    values[1025] = -1;

    auto* input = popnet_empty(popnet_dtype_i8, shape);
    REQUIRE(input != nullptr);
    auto* packed = popnet_cast(input, popnet_dtype_ternary);
    REQUIRE(packed != nullptr);
    auto* output = popnet_cast(packed, popnet_dtype_i8);
    REQUIRE(output != nullptr);
    CHECK(std::ranges::equal(popnet_shapeof(packed).dim, shape.dim));
    CHECK(popnet_dtypeof(packed) == popnet_dtype_ternary);
    CHECK(std::ranges::equal(popnet_shapeof(output).dim, shape.dim));
    CHECK(popnet_dtypeof(output) == popnet_dtype_i8);
    auto* graph = popnet_graph_alloc();
    REQUIRE(graph != nullptr);
    REQUIRE(popnet_graph_build_forward(graph, output));

    popnet_set_data(input, values.data(), 0, values.size());
    const int8_t patch = 1;
    popnet_set_data(input, &patch, 513, 1);
    popnet_graph_compute_forward(graph);

    int8_t host[3]{};
    popnet_get_data(input, host, 511, 3);
    CHECK(host[0] == -1);
    CHECK(host[1] == 1);
    CHECK(host[2] == 1);
    values[513] = patch;
    std::vector<int8_t> actual(values.size());
    popnet_get_data(output, actual.data(), 0, actual.size());
    CHECK(std::equal(actual.begin(), actual.end(), values.begin()));

    popnet_graph_free(graph);
    popnet_free(output);
    popnet_free(packed);
    popnet_free(input);
  }

  SUBCASE("f32 to i8 and back")
  {
    const float values[] = {-128.75f, -1.75f, -0.5f, 0.5f, 1.75f, 127.75f, 20, -10, 3};
    const int8_t expected[] = {-128, -1, 0, 0, 1, 127, 20, -10, 3};
    const popnet_shape shape{{1, 1, 1, 9}};
    auto* input = popnet_empty(popnet_dtype_f32, shape);
    REQUIRE(input != nullptr);
    auto* narrow = popnet_cast(input, popnet_dtype_i8);
    REQUIRE(narrow != nullptr);
    auto* wide = popnet_cast(narrow, popnet_dtype_f32);
    REQUIRE(wide != nullptr);
    CHECK(std::ranges::equal(popnet_shapeof(narrow).dim, shape.dim));
    CHECK(popnet_dtypeof(narrow) == popnet_dtype_i8);
    CHECK(std::ranges::equal(popnet_shapeof(wide).dim, shape.dim));
    CHECK(popnet_dtypeof(wide) == popnet_dtype_f32);
    auto* graph = popnet_graph_alloc();
    REQUIRE(graph != nullptr);
    REQUIRE(popnet_graph_build_forward(graph, wide));

    popnet_set_data(input, values, 0, 9);
    popnet_graph_compute_forward(graph);

    int8_t actual_i8[9]{};
    float actual_f32[9]{};
    popnet_get_data(narrow, actual_i8, 0, 9);
    popnet_get_data(wide, actual_f32, 0, 9);
    CHECK(std::equal(expected, expected + 9, actual_i8));
    CHECK(std::equal(expected, expected + 9, actual_f32));

    popnet_graph_free(graph);
    popnet_free(wide);
    popnet_free(narrow);
    popnet_free(input);
  }
}
