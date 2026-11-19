#include "elemwise.h"

#include <algorithm>
#include <cmath>
#include <doctest/doctest.h>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vector>

#include "dtype_traits.h"
#include "ops.h"
#include "tensor.h"
#include "test_assertions.h"
#include "test_data.h"

// Standard double libm references provide ample precision for these F32 tolerances;
// these sampled checks are not an exhaustive certification of libm or the domain.
template <auto compute> static void check_unary_accuracy(float actual, double expected)
{
  if constexpr (compute == popnet_compute_exp || compute == popnet_compute_log || compute == popnet_compute_sin ||
                compute == popnet_compute_cos || compute == popnet_compute_tanh || compute == popnet_compute_sigmoid)
    CHECK_F32_CLOSE(actual, expected, f32_math_atol, f32_math_rtol);
  else
    CHECK(actual == static_cast<float>(expected));
}

// @verifies REQ-0001: Compares unary results with scalar references.
// @verifies REQ-0007: Exercises float32 unary operations.
// @verifies REQ-0026: Correct rounding and specified mathematical-function tolerances.
TEST_CASE("elementwise unary compute")
{
  using namespace std::numbers;

  auto run = []<auto compute, auto compute_ref, popnet_dtype dtype>(auto lower, auto upper, bool exclude_zero = false) {
    auto run_n = [=](size_t n) {
      popnet_shape shape{{1, 1, 1, n}};

      auto input_data = generate_scalar_data<dtype>(n, lower, upper, exclude_zero);

      std::vector<double> expected_data;
      expected_data.reserve(n);

      for (auto v : input_data)
      {
        expected_data.push_back(compute_ref(v));
      }

      popnet_tensor* src = popnet_empty(dtype, shape);
      popnet_tensor* dst = popnet_empty(dtype, shape);

      REQUIRE(src != nullptr);
      REQUIRE(dst != nullptr);

      popnet_set_data(src, input_data.data(), dtype, 0, n);

      dst->src[0] = src;

      compute(dst);

      decltype(input_data) output_data;
      output_data.resize(n);

      popnet_get_data(dst, output_data.data(), dtype, 0, n);

      for (int i = 0; i < n; i++)
      {
        const auto actual = output_data[i];
        const auto expected = expected_data[i];

        CAPTURE(i);
        CAPTURE(input_data[i]);
        CAPTURE(actual);
        CAPTURE(expected);
        check_unary_accuracy<compute>(actual, expected);
      }

      popnet_free(src);
      popnet_free(dst);
    };

    SUBCASE("1") { run_n(1); }
    SUBCASE("2") { run_n(2); }
    SUBCASE("3") { run_n(3); }
    SUBCASE("4") { run_n(4); }
    SUBCASE("5") { run_n(5); }
    SUBCASE("15") { run_n(15); }
    SUBCASE("16") { run_n(16); }
    SUBCASE("17") { run_n(17); }
    SUBCASE("31") { run_n(31); }
    SUBCASE("32") { run_n(32); }
    SUBCASE("33") { run_n(33); }
    SUBCASE("1000") { run_n(1000); }
  };

  SUBCASE("abs - F32")
  {
    run.operator()<popnet_compute_abs, [](double v) { return std::abs(v); }, popnet_dtype_f32>(-10.0f, 10.0f);
  }

  SUBCASE("neg - F32")
  {
    run.operator()<popnet_compute_neg, [](double v) { return -v; }, popnet_dtype_f32>(-10.0f, 10.0f);
  }

  SUBCASE("sqr - F32")
  {
    run.operator()<popnet_compute_sqr, [](double v) { return v * v; }, popnet_dtype_f32>(-10.0f, 10.0f);
  }

  SUBCASE("sqrt - F32")
  {
    run.operator()<popnet_compute_sqrt, [](double v) { return std::sqrt(v); }, popnet_dtype_f32>(0.0f, 100.0f);
  }

  SUBCASE("exp - F32")
  {
    run.operator()<popnet_compute_exp, [](double v) { return std::exp(v); }, popnet_dtype_f32>(-10.0f, 10.0f);
  }

  SUBCASE("log - F32")
  {
    run.operator()<popnet_compute_log, [](double v) { return std::log(v); }, popnet_dtype_f32>(0.001f, 100.0f);
  }

  SUBCASE("sin - F32")
  {
    run.operator()<popnet_compute_sin, [](double v) { return std::sin(v); }, popnet_dtype_f32>(-2.0f * pi_v<float>,
                                                                                               2.0f * pi_v<float>);
  }

  SUBCASE("cos - F32")
  {
    run.operator()<popnet_compute_cos, [](double v) { return std::cos(v); }, popnet_dtype_f32>(-2.0f * pi_v<float>,
                                                                                               2.0f * pi_v<float>);
  }

  SUBCASE("tanh - F32")
  {
    run.operator()<popnet_compute_tanh, [](double v) { return std::tanh(v); }, popnet_dtype_f32>(-10.0f, 10.0f);
  }

  SUBCASE("relu - F32")
  {
    run.operator()<popnet_compute_relu, [](double v) { return (v > 0.f) ? v : 0.f; }, popnet_dtype_f32>(-10.0f, 10.0f);
  }

  SUBCASE("sigmoid - F32")
  {
    run.operator()<popnet_compute_sigmoid, [](double v) { return 1.f / (1.f + std::exp(-v)); }, popnet_dtype_f32>(
        -10.0f, 10.0f);
  }
}

size_t broadcast_index(size_t flat_idx, const popnet_shape& out_shape, const popnet_shape& in_shape)
{
  size_t in_flat = 0;
  size_t stride = 1;

  // walk dims from last to first (right-aligned)
  for (int d = POPNET_MAX_DIMS - 1; d >= 0; --d)
  {
    size_t out_dim = out_shape.dim[d];
    size_t in_dim = in_shape.dim[d];

    size_t coord = flat_idx % out_dim;
    flat_idx /= out_dim;

    // broadcast if input dim == 1
    size_t in_coord = (in_dim == 1 ? 0 : coord);

    in_flat += in_coord * stride;
    stride *= in_dim;
  }

  return in_flat;
}

template <popnet_dtype dtype, auto compute_ref>
static auto binary_ref(const auto& lhs, const popnet_shape& lhs_shape, const auto& rhs, const popnet_shape& rhs_shape,
                       const popnet_shape& out_shape)
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = typename dtype_traits::scalar_t;

  const size_t n = popnet_numel(out_shape);

  std::vector<scalar_t> out(n);

  for (size_t i = 0; i < n; ++i)
  {
    const size_t il = broadcast_index(i, out_shape, lhs_shape);
    const size_t ir = broadcast_index(i, out_shape, rhs_shape);

    out[i] = compute_ref(lhs[il], rhs[ir]);
  }

  return out;
}

// @verifies REQ-0002: Compares binary results across matching and broadcast shapes.
// @verifies REQ-0007: Exercises float32 binary operations.
// @verifies REQ-0026: Basic arithmetic rounds once to F32.
TEST_CASE("element binary compute")
{
  auto run = []<auto compute, auto compute_ref, popnet_dtype dtype>(auto lower, auto upper, bool exclude_zero = false) {
    using dtype_traits = dtype_traits<dtype>;
    using scalar_t = typename dtype_traits::scalar_t;

    auto run_shape = [=](const popnet_shape lhs_shape, const popnet_shape rhs_shape) {
      const auto dst_shape = popnet_broadcast_shape(lhs_shape, rhs_shape);
      const popnet_shape expected_shape{
          {std::max(lhs_shape.dim[0], rhs_shape.dim[0]), std::max(lhs_shape.dim[1], rhs_shape.dim[1]),
           std::max(lhs_shape.dim[2], rhs_shape.dim[2]), std::max(lhs_shape.dim[3], rhs_shape.dim[3])}};
      REQUIRE(dst_shape == expected_shape);

      const auto lhs_n = popnet_numel(lhs_shape);
      const auto rhs_n = popnet_numel(rhs_shape);
      const auto dst_n = popnet_numel(dst_shape);

      const auto lhs_data = generate_scalar_data<dtype>(lhs_n, lower, upper, exclude_zero, 0);
      const auto rhs_data = generate_scalar_data<dtype>(rhs_n, lower, upper, exclude_zero, 1);

      const auto expected_data = binary_ref<dtype, compute_ref>(lhs_data, lhs_shape, rhs_data, rhs_shape, dst_shape);

      popnet_tensor* lhs = popnet_empty(dtype, lhs_shape);
      popnet_tensor* rhs = popnet_empty(dtype, rhs_shape);
      popnet_tensor* dst = popnet_empty(dtype, dst_shape);

      REQUIRE(lhs != nullptr);
      REQUIRE(rhs != nullptr);
      REQUIRE(dst != nullptr);

      popnet_set_data(lhs, lhs_data.data(), dtype, 0, lhs_n);
      popnet_set_data(rhs, rhs_data.data(), dtype, 0, rhs_n);

      dst->src[0] = lhs;
      dst->src[1] = rhs;

      compute(dst);

      std::vector<scalar_t> output_data(dst_n);

      popnet_get_data(dst, output_data.data(), dtype, 0, dst_n);

      for (size_t i = 0; i < dst_n; ++i)
      {
        const auto actual = output_data[i];
        const auto expected = expected_data[i];

        CHECK(actual == expected);
      }

      popnet_free(lhs);
      popnet_free(rhs);
      popnet_free(dst);
    };

    SUBCASE("SIMD boundaries")
    {
      for (size_t n : {1, 3, 15, 16, 17, 31, 32, 33})
      {
        CAPTURE(n);

        run_shape(popnet_shape{{1, 1, 1, n}}, popnet_shape{{1, 1, 1, n}});
        run_shape(popnet_shape{{1, 1, 1, 1}}, popnet_shape{{1, 1, 1, n}});
        run_shape(popnet_shape{{1, 1, 1, n}}, popnet_shape{{1, 1, 1, 1}});
        run_shape(popnet_shape{{1, 1, 3, n}}, popnet_shape{{1, 1, 1, n}});
        run_shape(popnet_shape{{1, 1, 3, n}}, popnet_shape{{1, 1, 3, 1}});
      }
    }

    SUBCASE("1x1x1x1 1x1x1x1") { run_shape(popnet_shape{{1, 1, 1, 1}}, popnet_shape{{1, 1, 1, 1}}); }
    SUBCASE("1x1x1x2 1x1x1x2") { run_shape(popnet_shape{{1, 1, 1, 2}}, popnet_shape{{1, 1, 1, 2}}); }
    SUBCASE("1x1x1x3 1x1x1x3") { run_shape(popnet_shape{{1, 1, 1, 3}}, popnet_shape{{1, 1, 1, 3}}); }
    SUBCASE("1x1x1x4 1x1x1x4") { run_shape(popnet_shape{{1, 1, 1, 4}}, popnet_shape{{1, 1, 1, 4}}); }
    SUBCASE("1x1x1x5 1x1x1x5") { run_shape(popnet_shape{{1, 1, 1, 5}}, popnet_shape{{1, 1, 1, 5}}); }
    SUBCASE("1x1x1x1 1x1x1x8") { run_shape(popnet_shape{{1, 1, 1, 1}}, popnet_shape{{1, 1, 1, 8}}); }
    SUBCASE("1x1x1x16 1x1x1x1") { run_shape(popnet_shape{{1, 1, 1, 16}}, popnet_shape{{1, 1, 1, 1}}); }
    SUBCASE("1x1x4x5 1x3x1x5") { run_shape(popnet_shape{{1, 1, 4, 5}}, popnet_shape{{1, 3, 1, 5}}); }
    SUBCASE("1x2x1x7 1x1x9x7") { run_shape(popnet_shape{{1, 2, 1, 7}}, popnet_shape{{1, 1, 9, 7}}); }
    SUBCASE("1x1x4x1 1x3x1x8") { run_shape(popnet_shape{{1, 1, 4, 1}}, popnet_shape{{1, 3, 1, 8}}); }
    SUBCASE("1x2x1x1 1x1x5x6") { run_shape(popnet_shape{{1, 2, 1, 1}}, popnet_shape{{1, 1, 5, 6}}); }
    SUBCASE("1x1x7x1 1x8x1x9") { run_shape(popnet_shape{{1, 1, 7, 1}}, popnet_shape{{1, 8, 1, 9}}); }
    SUBCASE("1x4x1x9 1x1x6x1") { run_shape(popnet_shape{{1, 4, 1, 9}}, popnet_shape{{1, 1, 6, 1}}); }
    SUBCASE("1x3x1x5 4x1x6x1") { run_shape(popnet_shape{{1, 3, 1, 5}}, popnet_shape{{4, 1, 6, 1}}); }
    SUBCASE("2x1x7x1 1x5x1x9") { run_shape(popnet_shape{{2, 1, 7, 1}}, popnet_shape{{1, 5, 1, 9}}); }
    SUBCASE("1x1x1x1000 1x1x1x1000") { run_shape(popnet_shape{{1, 1, 1, 1000}}, popnet_shape{{1, 1, 1, 1000}}); }
  };

  SUBCASE("add - F32")
  {
    run.operator()<popnet_compute_add, [](double a, double b) { return a + b; }, popnet_dtype_f32>(-100.0f, 100.0f);
  }

  SUBCASE("sub - F32")
  {
    run.operator()<popnet_compute_sub, [](double a, double b) { return a - b; }, popnet_dtype_f32>(-100.0f, 100.0f);
  }

  SUBCASE("mul - F32")
  {
    run.operator()<popnet_compute_mul, [](double a, double b) { return a * b; }, popnet_dtype_f32>(-10.0f, 10.0f);
  }

  SUBCASE("div - F32")
  {
    run.operator()<popnet_compute_div, [](double a, double b) { return a / b; }, popnet_dtype_f32>(-10.0f, 10.0f, true);
  }
}

// @verifies REQ-0001: Unary finite-domain boundaries.
// @verifies REQ-0007: Uses the same F32 cases for every backend.
// @verifies REQ-0026: Correct rounding and finite mathematical-function tolerances.
TEST_CASE("unary F32 numerical boundaries")
{
  const float inf = std::numeric_limits<float>::infinity();
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float normal = std::numeric_limits<float>::min();
  const float largest = std::numeric_limits<float>::max();
  const std::vector<float> inputs = {0.0f,
                                     -0.0f,
                                     tiny,
                                     -tiny,
                                     normal,
                                     -normal,
                                     0x1p-12f,
                                     -0x1p-12f,
                                     std::nextafter(1.0f, 0.0f),
                                     1.0f,
                                     std::nextafter(1.0f, inf),
                                     -1.0f,
                                     0.056f,
                                     2.5f,
                                     std::nextafter(2.5f, inf),
                                     -2.5f,
                                     std::numbers::pi_v<float>,
                                     std::nextafter(std::numbers::pi_v<float>, inf),
                                     -104.0f,
                                     -103.972084f,
                                     -88.0f,
                                     88.0f,
                                     std::nextafter(88.722839f, 0.0f),
                                     88.722839f,
                                     1e20f,
                                     -1e20f,
                                     largest,
                                     -largest};

  auto run = [&]<auto compute>(auto reference) {
    for (float value : inputs)
      for (const size_t n : {1, 15, 16, 17})
      {
        CAPTURE(value);
        CAPTURE(n);
        const double expected = reference(double(value));
        // The accuracy contract covers real-domain inputs with finite F32 results.
        if (!std::isfinite(static_cast<float>(expected))) continue;
        std::vector<float> input(n, value), output(n);
        const popnet_shape shape{{1, 1, 1, n}};
        popnet_tensor src{.shape = shape, .data = input.data(), .dtype = popnet_dtype_f32};
        popnet_tensor dst{.shape = shape, .data = output.data(), .src = {&src}, .dtype = popnet_dtype_f32};

        compute(&dst);

        for (float actual : output)
        {
          CAPTURE(actual);
          CAPTURE(expected);
          check_unary_accuracy<compute>(actual, expected);
        }
      }
  };

  SUBCASE("abs")
  {
    run.operator()<popnet_compute_abs>([](double x) { return std::abs(x); });
  }
  SUBCASE("neg")
  {
    run.operator()<popnet_compute_neg>([](double x) { return -x; });
  }
  SUBCASE("sqr")
  {
    run.operator()<popnet_compute_sqr>([](double x) { return x * x; });
  }
  SUBCASE("sqrt")
  {
    run.operator()<popnet_compute_sqrt>([](double x) { return std::sqrt(x); });
  }
  SUBCASE("exp")
  {
    run.operator()<popnet_compute_exp>([](double x) { return std::exp(x); });
  }
  SUBCASE("log")
  {
    run.operator()<popnet_compute_log>([](double x) { return std::log(x); });
  }
  SUBCASE("sin")
  {
    run.operator()<popnet_compute_sin>([](double x) { return std::sin(x); });
  }
  SUBCASE("cos")
  {
    run.operator()<popnet_compute_cos>([](double x) { return std::cos(x); });
  }
  SUBCASE("tanh")
  {
    run.operator()<popnet_compute_tanh>([](double x) { return std::tanh(x); });
  }
  SUBCASE("relu")
  {
    run.operator()<popnet_compute_relu>([](double x) { return std::max(x, 0.0); });
  }
  SUBCASE("sigmoid")
  {
    // Equivalent mathematical definition, without overflowing exp(-x) in the reference.
    run.operator()<popnet_compute_sigmoid>([](double x) {
      const double e = std::exp(-std::abs(x));
      return x >= 0 ? 1 / (1 + e) : e / (1 + e);
    });
  }
}

// @verifies REQ-0002: Basic arithmetic at rounding and domain boundaries.
// @verifies REQ-0026: Correct rounding for finite results, including ties and subnormals.
TEST_CASE("binary F32 numerical boundaries")
{
  const float inf = std::numeric_limits<float>::infinity();
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float normal = std::numeric_limits<float>::min();
  const float largest = std::numeric_limits<float>::max();
  const std::pair<float, float> pairs[] = {{0.0f, -0.0f},    {-0.0f, -0.0f},
                                           {-0.0f, 2.0f},    {1.0f, 0.0f},
                                           {-1.0f, 0.0f},    {0.0f, 0.0f},
                                           {1.0f, 0x1p-24f}, {std::nextafter(1.0f, inf), 0x1p-24f},
                                           {1.0f, -1.0f},    {tiny, 0.5f},
                                           {3 * tiny, 0.5f}, {normal, 0.5f},
                                           {-normal, 0.5f},  {normal, std::nextafter(normal, 0.0f)},
                                           {largest, 2.0f},  {largest, largest},
                                           {1.0f, 3.0f}};

  auto run = [&]<auto compute>(auto reference) {
    for (const auto [a, b] : pairs)
    {
      CAPTURE(a);
      CAPTURE(b);
      // F32 products are exact in double; these sum/division fixtures avoid
      // double-rounding ambiguities and exercise ties with explicit dyadic inputs.
      const float expected = static_cast<float>(reference(double(a), double(b)));
      if (!std::isfinite(expected)) continue;
      std::vector<float> left(17, a), right(17, b), output(17);
      const popnet_shape shape{{1, 1, 1, 17}};
      popnet_tensor lhs{.shape = shape, .data = left.data(), .dtype = popnet_dtype_f32};
      popnet_tensor rhs{.shape = shape, .data = right.data(), .dtype = popnet_dtype_f32};
      popnet_tensor dst{.shape = shape, .data = output.data(), .src = {&lhs, &rhs}, .dtype = popnet_dtype_f32};

      compute(&dst);

      for (float actual : output)
      {
        CAPTURE(actual);
        CAPTURE(expected);
        CHECK(actual == expected);
      }
    }
  };

  SUBCASE("add")
  {
    run.operator()<popnet_compute_add>([](double a, double b) { return a + b; });
  }
  SUBCASE("sub")
  {
    run.operator()<popnet_compute_sub>([](double a, double b) { return a - b; });
  }
  SUBCASE("mul")
  {
    run.operator()<popnet_compute_mul>([](double a, double b) { return a * b; });
  }
  SUBCASE("div")
  {
    run.operator()<popnet_compute_div>([](double a, double b) { return a / b; });
  }
}

// @verifies REQ-0026: Numerical comparisons retain reference precision and reject nonfinite mismatches.
TEST_CASE("F32 comparisons retain double reference precision")
{
  const double reference = 1.0 + 0x1p-30;

  CHECK_FALSE(f32_close(1.0f, reference, 0, 0));
  CHECK(f32_close(1.0f, reference, 0x1p-30, 0));
  CHECK_F32_CLOSE(1.0f, reference, 0x1p-30, 0);
  CHECK_FALSE(f32_close(std::numeric_limits<float>::infinity(), reference, f32_math_atol, f32_math_rtol));
  CHECK_FALSE(f32_close(1.0f, std::numeric_limits<double>::quiet_NaN(), f32_math_atol, f32_math_rtol));
}

// @verifies REQ-0026: Sigmoid accuracy across its transition and exponential thresholds.
TEST_CASE("sigmoid F32 accuracy sweep")
{
  constexpr size_t n = 4097;
  std::vector<float> input(n), output(n);
  for (size_t i = 0; i < n; ++i)
    input[i] = static_cast<float>(-100.0 + 200.0 * double(i) / double(n - 1));

  const popnet_shape shape{{1, 1, 1, n}};
  popnet_tensor src{.shape = shape, .data = input.data(), .dtype = popnet_dtype_f32};
  popnet_tensor dst{.shape = shape, .data = output.data(), .src = {&src}, .dtype = popnet_dtype_f32};

  popnet_compute_sigmoid(&dst);

  for (size_t i = 0; i < n; ++i)
  {
    const double e = std::exp(-std::abs(double(input[i])));
    const double expected = input[i] < 0 ? e / (1 + e) : 1 / (1 + e);
    CAPTURE(i);
    CAPTURE(input[i]);
    CHECK_F32_CLOSE(output[i], expected, f32_math_atol, f32_math_rtol);
  }
}
