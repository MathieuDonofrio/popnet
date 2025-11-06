#include "elemwise.h"

#include <doctest/doctest.h>
#include <vector>

#include "simd.h"
#include "tensor.h"

template <popnet_type DType> static auto generate_data(size_t n)
{
  std::vector<typename simd_traits<DType>::scalar_t> data;
  data.reserve(n);

  switch (DType)
  {
  case POPNET_TYPE_F32:
  {
    for (size_t i = 0; i < n; ++i)
      data.push_back(static_cast<float>(i + 1) * (i % 2 == 0 ? 1.f : -1.f));
    break;
  }
  default: FAIL_CHECK("Unsupported dtype in generate_data"); break;
  }

  return data;
}

TEST_CASE("elementwise unary compute")
{
  auto run_for_compute = []<auto Compute, auto ScalarCompute, popnet_type DType>() {
    auto run_for_n = [](size_t n) {
      popnet_shape shape{{1, 1, 1, n}};

      auto input_data = generate_data<DType>(n);

      decltype(input_data) expected_data;
      expected_data.reserve(n);

      for (auto v : input_data)
      {
        expected_data.push_back(ScalarCompute(v));
      }

      popnet_tensor* src = popnet_empty(DType, shape);
      popnet_tensor* dst = popnet_empty(DType, shape);

      REQUIRE(src != nullptr);
      REQUIRE(dst != nullptr);

      REQUIRE(popnet_set_data(src, input_data.data(), DType, 0, n));

      dst->src[0] = src;

      Compute(dst);

      decltype(expected_data) output_data;
      output_data.resize(n);

      REQUIRE(popnet_get_data(dst, output_data.data(), DType, 0, n));

      for (int i = 0; i < n; i++)
      {
        const auto out = output_data[i];
        const auto expected = expected_data[i];

        if (std::isnan(expected))
        {
          CHECK(std::isnan(out));
        }
        else if (std::isinf(expected))
        {
          CHECK(std::isinf(out));
          CHECK((out > 0) == (expected > 0)); // same sign of infinity
        }
        else
        {
          CHECK(out == doctest::Approx(expected).epsilon(1e-4));
        }
      }

      popnet_free(src);
      popnet_free(dst);
    };

    SUBCASE("1") { run_for_n(1); }
    SUBCASE("simd length") { run_for_n(simd_traits<DType>::length()); }
    SUBCASE("simd length + 1") { run_for_n(simd_traits<DType>::length() + 1); }
    SUBCASE("simd length * 16") { run_for_n(simd_traits<DType>::length() * 16 + 1); }
    SUBCASE("simd length * 16 + 1") { run_for_n(simd_traits<DType>::length() * 16 + 1); }
  };

  SUBCASE("abs - F32")
  {
    run_for_compute.operator()<popnet_compute_abs, [](float v) { return std::abs(v); }, POPNET_TYPE_F32>();
  }

  SUBCASE("neg - F32")
  {
    run_for_compute.operator()<popnet_compute_neg, [](float v) { return -v; }, POPNET_TYPE_F32>();
  }

  SUBCASE("sqr - F32")
  {
    run_for_compute.operator()<popnet_compute_sqr, [](float v) { return v * v; }, POPNET_TYPE_F32>();
  }

  SUBCASE("sqrt - F32")
  {
    run_for_compute.operator()<popnet_compute_sqrt, [](float v) { return std::sqrt(v); }, POPNET_TYPE_F32>();
  }

  SUBCASE("exp - F32")
  {
    run_for_compute.operator()<popnet_compute_exp, [](float v) { return std::exp(v); }, POPNET_TYPE_F32>();
  }

  SUBCASE("log - F32")
  {
    run_for_compute.operator()<popnet_compute_log, [](float v) { return std::log(v); }, POPNET_TYPE_F32>();
  }

  SUBCASE("sin - F32")
  {
    run_for_compute.operator()<popnet_compute_sin, [](float v) { return std::sin(v); }, POPNET_TYPE_F32>();
  }

  SUBCASE("cos - F32")
  {
    run_for_compute.operator()<popnet_compute_cos, [](float v) { return std::cos(v); }, POPNET_TYPE_F32>();
  }

  SUBCASE("tanh - F32")
  {
    run_for_compute.operator()<popnet_compute_tanh, [](float v) { return std::tanh(v); }, POPNET_TYPE_F32>();
  }

  SUBCASE("relu - F32")
  {
    run_for_compute.operator()<popnet_compute_relu, [](float v) { return v > 0.f ? v : 0.f; }, POPNET_TYPE_F32>();
  }

  SUBCASE("sigmoid - F32")
  {
    run_for_compute
        .operator()<popnet_compute_sigmoid, [](float v) { return 1.f / (1.f + std::exp(-v)); }, POPNET_TYPE_F32>();
  }
}

TEST_CASE("element binary compute")
{
  auto run_for_binary = []<auto Compute, auto ScalarCompute, popnet_type DType>() {
    auto run_for_n = [](size_t n) {
      popnet_shape shape{{1, 1, 1, n}};

      auto lhs_data = generate_data<DType>(n);
      auto rhs_data = generate_data<DType>(n);

      decltype(lhs_data) expected_data;
      expected_data.reserve(n);

      for (size_t i = 0; i < n; ++i)
      {
        expected_data.push_back(ScalarCompute(lhs_data[i], rhs_data[i]));
      }

      popnet_tensor* lhs = popnet_empty(DType, shape);
      popnet_tensor* rhs = popnet_empty(DType, shape);
      popnet_tensor* dst = popnet_empty(DType, shape);

      REQUIRE(lhs != nullptr);
      REQUIRE(rhs != nullptr);
      REQUIRE(dst != nullptr);

      REQUIRE(popnet_set_data(lhs, lhs_data.data(), DType, 0, n));
      REQUIRE(popnet_set_data(rhs, rhs_data.data(), DType, 0, n));

      dst->src[0] = lhs;
      dst->src[1] = rhs;

      Compute(dst);

      decltype(lhs_data) output_data(n);
      REQUIRE(popnet_get_data(dst, output_data.data(), DType, 0, n));

      for (size_t i = 0; i < n; ++i)
      {
        const auto out = output_data[i];
        const auto expected = expected_data[i];

        if (std::isnan(expected))
        {
          CHECK(std::isnan(out));
        }
        else if (std::isinf(expected))
        {
          CHECK(std::isinf(out));
          CHECK((out > 0) == (expected > 0));
        }
        else
        {
          CHECK(out == doctest::Approx(expected).epsilon(1e-4));
        }
      }

      popnet_free(lhs);
      popnet_free(rhs);
      popnet_free(dst);
    };

    SUBCASE("1") { run_for_n(1); }
    SUBCASE("simd elements") { run_for_n(simd_traits<DType>::length()); }
    SUBCASE("simd elements + 1") { run_for_n(simd_traits<DType>::length() + 1); }
    SUBCASE("simd elements * 16") { run_for_n(simd_traits<DType>::length() * 16); }
    SUBCASE("simd elements * 16 + 1") { run_for_n(simd_traits<DType>::length() * 16 + 1); }
  };

  SUBCASE("add - F32")
  {
    run_for_binary.operator()<popnet_compute_add, [](float a, float b) { return a + b; }, POPNET_TYPE_F32>();
  }

  SUBCASE("sub - F32")
  {
    run_for_binary.operator()<popnet_compute_sub, [](float a, float b) { return a - b; }, POPNET_TYPE_F32>();
  }

  SUBCASE("mul - F32")
  {
    run_for_binary.operator()<popnet_compute_mul, [](float a, float b) { return a * b; }, POPNET_TYPE_F32>();
  }

  SUBCASE("div - F32")
  {
    run_for_binary.operator()<popnet_compute_div, [](float a, float b) { return a / b; }, POPNET_TYPE_F32>();
  }
}
