#include "reduce.h"

#include <algorithm>
#include <cstring>
#include <doctest/doctest.h>
#include <limits>
#include <type_traits>
#include <vector>

#include "ops.h"
#include "simd/simd_traits.h"
#include "tensor.h"
#include "test_assertions.h"
#include "test_data.h"

template <popnet_dtype dtype, class InitFn, class MergeFn, class FinalizeFn>
static auto reduce_ref(const auto& src, popnet_shape src_shape, auto& dst, popnet_shape dst_shape, InitFn init_fn,
                       MergeFn merge_fn, FinalizeFn finalize_fn)
{
  const size_t src_numel = popnet_numel(src_shape);
  const size_t dst_numel = popnet_numel(dst_shape);

  dst.assign(dst_numel, init_fn());
  std::vector<size_t> counts(dst_numel, 0);

  const size_t src_strides[POPNET_MAX_DIMS] = {src_shape.dim[1] * src_shape.dim[2] * src_shape.dim[3],
                                               src_shape.dim[2] * src_shape.dim[3], src_shape.dim[3], 1};
  const size_t dst_strides[POPNET_MAX_DIMS] = {dst_shape.dim[1] * dst_shape.dim[2] * dst_shape.dim[3],
                                               dst_shape.dim[2] * dst_shape.dim[3], dst_shape.dim[3], 1};

  for (size_t d = 0; d < POPNET_MAX_DIMS; ++d)
  {
    if (dst_shape.dim[d] != 1)
    {
      POPNET_EXPECT(dst_shape.dim[d] == src_shape.dim[d]);
    }
  }

  for (size_t src_idx = 0; src_idx < src_numel; ++src_idx)
  {
    size_t tmp = src_idx;
    size_t src_coord[POPNET_MAX_DIMS] = {};

    for (size_t d = 0; d < POPNET_MAX_DIMS; ++d)
    {
      src_coord[d] = tmp / src_strides[d];
      tmp -= src_coord[d] * src_strides[d];
    }

    size_t dst_coord[POPNET_MAX_DIMS] = {};
    for (size_t d = 0; d < POPNET_MAX_DIMS; ++d)
    {
      dst_coord[d] = dst_shape.dim[d] == 1 ? 0 : src_coord[d];
    }

    size_t dst_idx = 0;
    for (size_t d = 0; d < POPNET_MAX_DIMS; ++d)
    {
      dst_idx += dst_coord[d] * dst_strides[d];
    }

    merge_fn(dst[dst_idx], src[src_idx]);
    ++counts[dst_idx];
  }

  for (size_t i = 0; i < dst_numel; ++i)
  {
    dst[i] = finalize_fn(dst[i], counts[i]);
  }

  return dst;
}

// Check each operation against the exact mathematical reference, reserving part
// of the F32 bound for binary64 reference rounding rather than loosening the bound.
static void check_reduction_accuracy(decltype(&popnet_compute_reduce_sum) compute, const std::vector<float>& input,
                                     popnet_shape shape, popnet_shape output_shape, const std::vector<float>& actual)
{
  const size_t count = input.size() / actual.size();
  if (count == 1)
  {
    CHECK(std::memcmp(actual.data(), input.data(), input.size() * sizeof(float)) == 0);
    return;
  }
  const bool product = compute == popnet_compute_reduce_product;
  const bool mean = compute == popnet_compute_reduce_mean;
  const bool minimum = compute == popnet_compute_reduce_min;
  const bool maximum = compute == popnet_compute_reduce_max;
  std::vector<double> expected, magnitudes;
  reduce_ref<popnet_dtype_f32>(
      input, shape, expected, output_shape,
      [&] {
        return product   ? 1.0
               : minimum ? std::numeric_limits<double>::infinity()
               : maximum ? -std::numeric_limits<double>::infinity()
                         : 0.0;
      },
      [&](double& acc, float value) {
        if (product) acc *= value;
        else if (minimum)
          acc = std::min(acc, double(value));
        else if (maximum)
          acc = std::max(acc, double(value));
        else
          acc += value;
      },
      [&](double acc, size_t n) { return mean ? acc / double(n) : acc; });
  reduce_ref<popnet_dtype_f32>(
      input, shape, magnitudes, output_shape, [] { return 0.0; },
      [](double& acc, float value) { acc += std::abs(double(value)); },
      [&](double acc, size_t n) { return mean ? acc / double(n) : acc; });

  for (size_t i = 0; i < actual.size(); ++i)
  {
    CAPTURE(i);
    CAPTURE(count);
    CAPTURE(actual[i]);
    CAPTURE(expected[i]);
    if (minimum || maximum)
    {
      CHECK(actual[i] == expected[i]);
      continue;
    }
    const size_t steps = mean ? count : count - 1;
    const double reference_gamma = rounding_gamma(steps, 0x1p-53);
    const double scale = product ? std::abs(expected[i]) : magnitudes[i];
    const double reference_error = reference_gamma * scale / (1 - reference_gamma);
    const double bound = rounding_gamma(steps) * scale / (1 + reference_gamma);
    REQUIRE(std::isfinite(expected[i]));
    CHECK_F32_CLOSE(actual[i], expected[i], bound - reference_error, 0);
  }
}

// @verifies REQ-0003: Checks singleton-axis reduction values, including special floating-point values.
// @verifies REQ-0007: Exercises float32 singleton-axis reductions.
// @verifies REQ-0026: Singleton reductions preserve every representation bit.
TEST_CASE("reductions over singleton axes preserve input values")
{
  const auto reductions = {popnet_compute_reduce_sum, popnet_compute_reduce_mean, popnet_compute_reduce_product,
                           popnet_compute_reduce_min, popnet_compute_reduce_max};
  const popnet_shape shapes[] = {{{2, 3, 1, 1}}, {{1, 3, 1, 2}}};
  const std::vector<float> input = {-0.0f,
                                    std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(),
                                    3.0f,
                                    -2.0f};

  for (auto compute : reductions)
  {
    // Scalar singleton values are copied bit-for-bit for every reduction.
    {
      for (float value : input)
      {
        float result = 0.0f;
        popnet_tensor src{.shape = popnet_scalar_shape, .data = &value, .dtype = popnet_dtype_f32};
        popnet_tensor dst{.shape = popnet_scalar_shape, .data = &result, .src = {&src}, .dtype = popnet_dtype_f32};

        compute(&dst);

        CHECK(std::memcmp(&result, &value, sizeof(float)) == 0);
      }
    }

    for (const auto shape : shapes)
    {
      auto* src = popnet_empty(popnet_dtype_f32, shape);
      REQUIRE(src != nullptr);
      popnet_set_data(src, input.data(), 0, input.size());
      auto* dst = popnet_empty(popnet_dtype_f32, shape);
      REQUIRE(dst != nullptr);
      CHECK(popnet_shapeof(dst) == shape);
      CHECK(dst->data != src->data);

      dst->src[0] = src;
      compute(dst);

      std::vector<float> actual(input.size());
      popnet_get_data(dst, actual.data(), 0, actual.size());
      CHECK(std::memcmp(actual.data(), input.data(), input.size() * sizeof(float)) == 0);

      popnet_free(dst);
      popnet_free(src);
    }
  }
}

// @verifies REQ-0003: Composed reductions over separated groups match the scalar reference.
// @verifies REQ-0025: Composed reductions leave the input unchanged.
// @verifies REQ-0026: Each operation in the composition satisfies its own accuracy bound.
TEST_CASE("separated reduction groups compose through the public API")
{
  struct reduction
  {
    decltype(&popnet_reduce_sum) create;
    decltype(&popnet_compute_reduce_sum) compute;
  };
  const reduction operations[] = {{popnet_reduce_sum, popnet_compute_reduce_sum},
                                  {popnet_reduce_mean, popnet_compute_reduce_mean},
                                  {popnet_reduce_product, popnet_compute_reduce_product},
                                  {popnet_reduce_min, popnet_compute_reduce_min},
                                  {popnet_reduce_max, popnet_compute_reduce_max}};
  const popnet_shape shape{{2, 3, 4, 5}};
  const popnet_shape expected_shape{{1, 3, 1, 1}};
  std::vector<float> input(popnet_numel(shape));
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = 1.0f + static_cast<float>(i % 17) / 100.0f;

  for (const auto& operation : operations)
  {
    auto* src = popnet_empty(popnet_dtype_f32, shape);
    REQUIRE(src != nullptr);
    popnet_set_data(src, input.data(), 0, input.size());
    auto* spatial = operation.create(src, 0x0C);
    REQUIRE(spatial != nullptr);
    auto* result = operation.create(spatial, 0x01);
    REQUIRE(result != nullptr);
    auto* graph = popnet_graph_alloc();
    REQUIRE(graph != nullptr);
    REQUIRE(popnet_graph_build_forward(graph, result));

    popnet_graph_compute_forward(graph);

    CHECK(popnet_shapeof(result) == expected_shape);
    std::vector<float> intermediate(popnet_numel(popnet_shapeof(spatial)));
    popnet_get_data(spatial, intermediate.data(), 0, intermediate.size());
    check_reduction_accuracy(operation.compute, input, shape, popnet_shapeof(spatial), intermediate);
    std::vector<float> actual(popnet_numel(expected_shape));
    popnet_get_data(result, actual.data(), 0, actual.size());
    check_reduction_accuracy(operation.compute, intermediate, popnet_shapeof(spatial), expected_shape, actual);
    std::vector<float> preserved(input.size());
    popnet_get_data(src, preserved.data(), 0, preserved.size());
    CHECK(std::memcmp(preserved.data(), input.data(), input.size() * sizeof(float)) == 0);

    popnet_graph_free(graph);
    popnet_free(result);
    popnet_free(spatial);
    popnet_free(src);
  }
}

// After removing singleton dimensions, a supported mask contains just one run of set bits.
static bool supported_mask(popnet_shape shape, uint8_t mask)
{
  unsigned bits = 0;
  for (size_t i = 0; i < POPNET_MAX_DIMS; ++i)
    if (shape.dim[i] != 1) bits = (bits << 1) | ((mask >> i) & 1);
  if (bits == 0) return true;
  while ((bits & 1) == 0)
    bits >>= 1;
  return (bits & (bits + 1)) == 0;
}

// @verifies REQ-0003: Compares reduction results with scalar references for supported axis groups.
// @verifies REQ-0007: Exercises all five float32 reduction operations.
// @verifies REQ-0025: Checks input preservation for supported reduction-axis masks.
// @verifies REQ-0026: Reduction errors scale with count and input magnitudes.
TEST_CASE("reduce compute")
{
  auto run = []<auto compute, popnet_dtype dtype>(auto lower, auto upper, bool exclude_zero = false) {
    auto run_shape = [&](popnet_shape src_shape, popnet_shape dst_shape) {
      using traits = simd_traits<dtype>;
      using scalar_t = typename traits::scalar_t;

      const size_t n = popnet_numel(src_shape);

      auto input_data = generate_scalar_data<dtype>(n, lower, upper, exclude_zero);

      popnet_tensor* src = popnet_empty(dtype, src_shape);
      popnet_tensor* dst = popnet_empty(dtype, dst_shape);

      REQUIRE(src != nullptr);
      REQUIRE(dst != nullptr);

      popnet_set_data(src, input_data.data(), 0, input_data.size());

      dst->src[0] = src;

      compute(dst);

      std::vector<scalar_t> preserved(n);
      popnet_get_data(src, preserved.data(), 0, n);
      CHECK(std::memcmp(preserved.data(), input_data.data(), n * sizeof(scalar_t)) == 0);
      CHECK(popnet_shapeof(dst) == dst_shape);

      std::vector<scalar_t> actual(popnet_numel(dst_shape));
      popnet_get_data(dst, actual.data(), 0, actual.size());

      check_reduction_accuracy(compute, input_data, src_shape, dst_shape, actual);

      popnet_free(src);
      popnet_free(dst);
    };

    SUBCASE("1x1x1x1 -> 1x1x1x1") { run_shape({{1, 1, 1, 1}}, popnet_scalar_shape); }
    SUBCASE("2x3x1x1 -> 2x3x1x1") { run_shape({{2, 3, 1, 1}}, {{2, 3, 1, 1}}); }
    SUBCASE("1x3x1x2 -> 1x3x1x2") { run_shape({{1, 3, 1, 2}}, {{1, 3, 1, 2}}); }
    SUBCASE("2x2x2x2 -> 1x1x1x1") { run_shape({{2, 2, 2, 2}}, popnet_scalar_shape); }
    SUBCASE("2x3x4x5 -> 1x1x1x1") { run_shape({{2, 3, 4, 5}}, popnet_scalar_shape); }
    SUBCASE("4x4x4x4 -> 1x1x1x1") { run_shape({{4, 4, 4, 4}}, popnet_scalar_shape); }
    SUBCASE("8x2x3x5 -> 1x1x1x1") { run_shape({{8, 2, 3, 5}}, popnet_scalar_shape); }
    SUBCASE("3x4x5x1 -> 1x1x1x1") { run_shape({{3, 4, 5, 1}}, popnet_scalar_shape); }
    SUBCASE("3x4x1x1 -> 1x1x1x1") { run_shape({{3, 4, 1, 1}}, popnet_scalar_shape); }
    SUBCASE("3x1x1x1 -> 1x1x1x1") { run_shape({{3, 1, 1, 1}}, popnet_scalar_shape); }
    SUBCASE("2x3x4x5 -> 2x3x4x1") { run_shape({{2, 3, 4, 5}}, {{2, 3, 4, 1}}); }
    SUBCASE("2x3x4x5 -> 2x3x1x1") { run_shape({{2, 3, 4, 5}}, {{2, 3, 1, 1}}); }
    SUBCASE("2x3x4x5 -> 2x1x1x1") { run_shape({{2, 3, 4, 5}}, {{2, 1, 1, 1}}); }
    SUBCASE("output vector widths and tails")
    {
      for (const size_t rows : {1, 15, 16, 17, 33})
      {
        CAPTURE(rows);
        run_shape({{1, rows, 3, 5}}, {{1, rows, 1, 1}});
      }
    }
    SUBCASE("contiguous reduction widths with multiple output groups")
    {
      for (const size_t count : {3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 31, 33, 63, 64, 65})
      {
        CAPTURE(count);
        run_shape({{2, 3, 1, count}}, {{2, 3, 1, 1}});
      }
    }
    SUBCASE("row reduction counts with multiple output groups")
    {
      for (const size_t count : {3, 16, 17, 33})
        for (const size_t inner : {3, 16, 17, 33})
        {
          CAPTURE(count);
          CAPTURE(inner);
          run_shape({{2, 1, count, inner}}, {{2, 1, 1, inner}});
        }
    }
    SUBCASE("all supported axis masks")
    {
      const popnet_shape shapes[] = {{{2, 3, 4, 5}}, {{1, 3, 4, 5}}, {{2, 1, 4, 5}},
                                     {{2, 3, 1, 5}}, {{2, 3, 4, 1}}, {{1, 1, 1, 1}}};
      for (const auto shape : shapes)
        for (uint8_t mask = 1; mask < 16; ++mask)
        {
          CAPTURE(mask);
          if (!supported_mask(shape, mask)) continue;
          run_shape(shape, popnet_reduce_shape(shape, mask));
        }
    }
    SUBCASE("row reductions and SIMD boundaries")
    {
      for (const size_t inner : {1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 31, 32, 33, 63, 64, 65})
        for (uint8_t mask = 1; mask < 16; ++mask)
        {
          CAPTURE(inner);
          CAPTURE(mask);
          const popnet_shape shape{{2, 3, 2, inner}};
          if (!supported_mask(shape, mask)) continue;
          run_shape(shape, popnet_reduce_shape(shape, mask));
        }
    }
  };

  SUBCASE("sum - F32") { run.operator()<popnet_compute_reduce_sum, popnet_dtype_f32>(-10.0f, 10.0f); }
  SUBCASE("mean - F32") { run.operator()<popnet_compute_reduce_mean, popnet_dtype_f32>(-10.0f, 10.0f); }
  SUBCASE("product - F32")
  {
    // Keep every possible partial product normal for all tested shapes/orders.
    run.operator()<popnet_compute_reduce_product, popnet_dtype_f32>(0.9f, 1.1f);
  }
  SUBCASE("min - F32") { run.operator()<popnet_compute_reduce_min, popnet_dtype_f32>(-100.0f, 100.0f); }
  SUBCASE("max - F32") { run.operator()<popnet_compute_reduce_max, popnet_dtype_f32>(-100.0f, 100.0f); }
}

// @verifies REQ-0003: Contiguous and strided reductions use the same numerical contract.
// @verifies REQ-0026: Cancellation, magnitude differences, product signs, and zeros.
TEST_CASE("reduction F32 numerical boundaries")
{
  const struct
  {
    const char* name;
    decltype(&popnet_compute_reduce_sum) compute;
  } operations[] = {{"sum", popnet_compute_reduce_sum},
                    {"mean", popnet_compute_reduce_mean},
                    {"prod", popnet_compute_reduce_product},
                    {"min", popnet_compute_reduce_min},
                    {"max", popnet_compute_reduce_max}};
  for (const auto& operation : operations)
    for (size_t count : {2, 3, 15, 16, 17, 31, 32, 33, 65})
      for (size_t inner : {1, 17})
      {
        INFO("operation: " << doctest::String(operation.name));
        CAPTURE(count);
        CAPTURE(inner);
        const auto compute = operation.compute;
        const popnet_shape shape{{1, 1, count, inner}}, result_shape{{1, 1, 1, inner}};
        std::vector<float> input(count * inner), output(inner);
        const float sum_pattern[] = {0x1p20f, 0x1p-10f, -0x1p20f, -0x1p-10f, 0x1p-12f};
        const float product_pattern[] = {-2, 0.5f, -1, 1};
        for (size_t row = 0; row < count; ++row)
          for (size_t col = 0; col < inner; ++col)
            input[row * inner + col] = compute == popnet_compute_reduce_product ? product_pattern[(row + col) % 4]
                                                                                : sum_pattern[(row + col) % 5];
        popnet_tensor src{.shape = shape, .data = input.data(), .dtype = popnet_dtype_f32};
        popnet_tensor dst{.shape = result_shape, .data = output.data(), .src = {&src}, .dtype = popnet_dtype_f32};

        compute(&dst);

        check_reduction_accuracy(compute, input, shape, result_shape, output);

        // These values have an order-independent result for every operation.
        for (float value : {0.0f, -0.0f})
        {
          std::fill(input.begin(), input.end(), value);

          compute(&dst);

          for (float actual : output)
            CHECK(actual == value);
        }
      }
}
