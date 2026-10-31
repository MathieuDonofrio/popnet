#include "reduce.h"

#include <cstring>
#include <doctest/doctest.h>
#include <limits>
#include <vector>

#include "dtype_traits.h"
#include "ops.h"
#include "tensor.h"
#include "test_utils.h"

template <popnet_dtype dtype, class InitFn, class MergeFn, class FinalizeFn>
static auto reduce_ref(const auto& src, popnet_shape src_shape, auto& dst, popnet_shape dst_shape, InitFn init_fn,
                       MergeFn merge_fn, FinalizeFn finalize_fn)
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = typename dtype_traits::scalar_t;

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

    merge_fn(dst[dst_idx], static_cast<scalar_t>(src[src_idx]));
    ++counts[dst_idx];
  }

  for (size_t i = 0; i < dst_numel; ++i)
  {
    dst[i] = finalize_fn(dst[i], counts[i]);
  }

  return dst;
}

// @verifies REQ-0003: Checks singleton-axis reduction values, including special floating-point values.
// @verifies REQ-0007: Exercises float32 singleton-axis reductions.
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
      popnet_set_data(src, input.data(), popnet_dtype_f32, 0, input.size());
      auto* dst = popnet_empty(popnet_dtype_f32, shape);
      REQUIRE(dst != nullptr);
      CHECK(popnet_shapeof(dst) == shape);
      CHECK(dst->data != src->data);

      dst->src[0] = src;
      compute(dst);

      std::vector<float> actual(input.size());
      popnet_get_data(dst, actual.data(), popnet_dtype_f32, 0, actual.size());
      CHECK(std::memcmp(actual.data(), input.data(), input.size() * sizeof(float)) == 0);

      popnet_free(dst);
      popnet_free(src);
    }
  }
}


// This regression records existing backend behavior; NaN/tie semantics for non-singleton min/max are not specified.
TEST_CASE("row min and max preserve existing scalar and SIMD operand semantics")
{
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  for (auto compute : {popnet_compute_reduce_min, popnet_compute_reduce_max})
  {
    std::vector<float> input(34, 3.0f);
    input[17] = nan;
    input[1] = +0.0f;
    input[18] = -0.0f;
    input[16] = +0.0f;
    input[33] = -0.0f;
    std::vector<float> output(17);
    popnet_tensor src{.shape = {{1, 1, 2, 17}}, .data = input.data(), .dtype = popnet_dtype_f32};
    popnet_tensor dst{.shape = {{1, 1, 1, 17}}, .data = output.data(), .src = {&src}, .dtype = popnet_dtype_f32};

    compute(&dst);

    if constexpr (dtype_traits<popnet_dtype_f32>::cpu_execution_model == cpu_execution_model::fixed)
    {
      CHECK(output[0] == 3.0f);
      CHECK_FALSE(std::signbit(output[1]));
      CHECK_FALSE(std::signbit(output[16]));
    }
    else
    {
      CHECK(std::isnan(output[0]));
      CHECK(std::signbit(output[1]));
      CHECK(std::signbit(output[16]));
    }
    CHECK(output[1] == 0.0f);
    CHECK(output[16] == 0.0f);
  }
}



// @verifies REQ-0003: Composed reductions over separated groups match the scalar reference.
// @verifies REQ-0025: Composed reductions leave the input unchanged.
TEST_CASE("separated reduction groups compose through the public API")
{
  struct reduction
  {
    decltype(&popnet_reduce_sum) create;
    float initial;
    float (*merge)(float, float);
    bool mean;
  };
  const reduction operations[] = {
      {popnet_reduce_sum, 0, [](float a, float b) { return a + b; }, false},
      {popnet_reduce_mean, 0, [](float a, float b) { return a + b; }, true},
      {popnet_reduce_product, 1, [](float a, float b) { return a * b; }, false},
      {popnet_reduce_min, std::numeric_limits<float>::infinity(),
       [](float a, float b) { return std::min(a, b); }, false},
      {popnet_reduce_max, -std::numeric_limits<float>::infinity(),
       [](float a, float b) { return std::max(a, b); }, false}};
  const popnet_shape shape{{2, 3, 4, 5}};
  const popnet_shape expected_shape{{1, 3, 1, 1}};
  std::vector<float> input(popnet_numel(shape));
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = 1.0f + static_cast<float>(i % 17) / 100.0f;

  for (const auto& operation : operations)
  {
    std::vector<float> expected;
    reduce_ref<popnet_dtype_f32>(
        input, shape, expected, expected_shape, [&] { return operation.initial; },
        [&](float& acc, float value) { acc = operation.merge(acc, value); },
        [&](float acc, size_t count) { return operation.mean ? acc / static_cast<float>(count) : acc; });
    auto* src = popnet_empty(popnet_dtype_f32, shape);
    REQUIRE(src != nullptr);
    popnet_set_data(src, input.data(), popnet_dtype_f32, 0, input.size());
    auto* spatial = operation.create(src, 0x0C);
    REQUIRE(spatial != nullptr);
    auto* result = operation.create(spatial, 0x01);
    REQUIRE(result != nullptr);
    auto* graph = popnet_graph_alloc();
    REQUIRE(graph != nullptr);
    REQUIRE(popnet_graph_build_forward(graph, result));

    popnet_graph_compute_forward(graph);

    CHECK(popnet_shapeof(result) == expected_shape);
    std::vector<float> actual(expected.size());
    popnet_get_data(result, actual.data(), popnet_dtype_f32, 0, actual.size());
    for (size_t i = 0; i < actual.size(); ++i)
      CHECK(actual[i] == doctest::Approx(expected[i]));
    std::vector<float> preserved(input.size());
    popnet_get_data(src, preserved.data(), popnet_dtype_f32, 0, preserved.size());
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
TEST_CASE("reduce compute")
{
  auto run = []<auto compute, auto compute_ref, popnet_dtype dtype>(auto lower, auto upper, bool exclude_zero = false) {
    auto run_shape = [&](popnet_shape src_shape, popnet_shape dst_shape) {
      using dtype_dtype_traits = dtype_traits<dtype>;
      using scalar_t = typename dtype_dtype_traits::scalar_t;

      const size_t n = popnet_numel(src_shape);

      auto input_data = generate_scalar_data<dtype>(n, lower, upper, exclude_zero);

      auto expected = compute_ref(input_data, src_shape, dst_shape);

      popnet_tensor* src = popnet_empty(dtype, src_shape);
      popnet_tensor* dst = popnet_empty(dtype, dst_shape);

      REQUIRE(src != nullptr);
      REQUIRE(dst != nullptr);

      popnet_set_data(src, input_data.data(), dtype, 0, input_data.size());

      dst->src[0] = src;

      compute(dst);

      std::vector<scalar_t> preserved(n);
      popnet_get_data(src, preserved.data(), dtype, 0, n);
      CHECK(std::memcmp(preserved.data(), input_data.data(), n * sizeof(scalar_t)) == 0);
      CHECK(popnet_shapeof(dst) == dst_shape);

      std::vector<scalar_t> actual(popnet_numel(dst_shape));
      popnet_get_data(dst, actual.data(), dtype, 0, actual.size());

      REQUIRE(actual.size() == expected.size());
      for (size_t i = 0; i < actual.size(); ++i)
      {
        if constexpr (std::is_floating_point_v<scalar_t>)
        {
          CHECK(actual[i] == doctest::Approx(expected[i]));
        }
        else
        {
          CHECK(actual[i] == expected[i]);
        }
      }

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
      for (const size_t inner : {1, 3, 15, 16, 17, 31, 32, 33})
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

  SUBCASE("sum - F32")
  {
    run.operator()<popnet_compute_reduce_sum,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_dtype_f32>(
                         src, src_shape, dst, dst_shape, [] { return 0.0f; }, [](float& acc, float v) { acc += v; },
                         [](float acc, size_t) { return acc; });
                   },
                   popnet_dtype_f32>(-10.0f, 10.0f);
  }

  SUBCASE("mean - F32")
  {
    run.operator()<popnet_compute_reduce_mean,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_dtype_f32>(
                         src, src_shape, dst, dst_shape, [] { return 0.0f; }, [](float& acc, float v) { acc += v; },
                         [](float acc, size_t count) { return acc / count; });
                   },
                   popnet_dtype_f32>(-10.0f, 10.0f);
  }

  SUBCASE("product - F32")
  {
    run.operator()<popnet_compute_reduce_product,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_dtype_f32>(
                         src, src_shape, dst, dst_shape, [] { return 1.0f; }, [](float& acc, float v) { acc *= v; },
                         [](float acc, size_t) { return acc; });
                   },
                   popnet_dtype_f32>(0.5f, 2.0f); // avoids zero underflow
  }

  SUBCASE("min - F32")
  {
    run.operator()<popnet_compute_reduce_min,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_dtype_f32>(
                         src, src_shape, dst, dst_shape, [] { return std::numeric_limits<float>::infinity(); },
                         [](float& acc, float v) { acc = std::min(acc, v); }, [](float acc, size_t) { return acc; });
                   },
                   popnet_dtype_f32>(-100.0f, 100.0f);
  }

  SUBCASE("max - F32")
  {
    run.operator()<popnet_compute_reduce_max,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_dtype_f32>(
                         src, src_shape, dst, dst_shape, [] { return -std::numeric_limits<float>::infinity(); },
                         [](float& acc, float v) { acc = std::max(acc, v); }, [](float acc, size_t) { return acc; });
                   },
                   popnet_dtype_f32>(-100.0f, 100.0f);
  }
}
