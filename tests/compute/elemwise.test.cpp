#include "elemwise.h"

#include <doctest/doctest.h>
#include <vector>

#include "ops.h"
#include "tensor.h"
#include "test_utils.h"
#include "traits.h"

TEST_CASE("elementwise unary compute")
{
  using namespace std::numbers;

  auto run =
      []<auto compute, auto compute_ref, popnet_type dtype, auto lower, auto upper, bool exclude_zero = false>() {
        auto run_n = [](size_t n) {
          popnet_shape shape{{1, 1, 1, n}};

          auto input_data = generate_scalar_data<dtype>(n, lower, upper, exclude_zero);

          decltype(input_data) expected_data;
          expected_data.reserve(n);

          for (auto v : input_data)
          {
            expected_data.push_back(compute_ref(v));
          }

          popnet_tensor* src = popnet_empty(dtype, shape);
          popnet_tensor* dst = popnet_empty(dtype, shape);

          REQUIRE(src != nullptr);
          REQUIRE(dst != nullptr);

          REQUIRE(popnet_set_data(src, input_data.data(), dtype, 0, n));

          dst->src[0] = src;

          compute(dst);

          decltype(expected_data) output_data;
          output_data.resize(n);

          REQUIRE(popnet_get_data(dst, output_data.data(), dtype, 0, n));

          for (int i = 0; i < n; i++)
          {
            const auto actual = output_data[i];
            const auto expected = expected_data[i];

            if constexpr (std::is_floating_point_v<decltype(expected)>)
            {
              CHECK(actual == doctest::Approx(expected).epsilon(1e-4));
            }
            else
            {
              CHECK(actual == expected);
            }
          }

          popnet_free(src);
          popnet_free(dst);
        };

        SUBCASE("1") { run_n(1); }
        SUBCASE("2") { run_n(2); }
        SUBCASE("3") { run_n(3); }
        SUBCASE("4") { run_n(4); }
        SUBCASE("5") { run_n(5); }
        SUBCASE("1000") { run_n(1000); }
        SUBCASE("simd") { run_n(traits<dtype>::vector_length()); }
        SUBCASE("simd + 1") { run_n(traits<dtype>::vector_length() + 1); }
        SUBCASE("simd * 16") { run_n(traits<dtype>::vector_length() * 16 + 1); }
        SUBCASE("simd * 16 + 1") { run_n(traits<dtype>::vector_length() * 16 + 1); }
      };

  SUBCASE("abs - F32")
  {
    run.operator()<popnet_compute_abs, [](float v) { return std::abs(v); }, POPNET_TYPE_F32, -10.0f, 10.0f>();
  }

  SUBCASE("neg - F32")
  {
    run.operator()<popnet_compute_neg, [](float v) { return -v; }, POPNET_TYPE_F32, -10.0f, 10.0f>();
  }

  SUBCASE("sqr - F32")
  {
    run.operator()<popnet_compute_sqr, [](float v) { return v * v; }, POPNET_TYPE_F32, -10.0f, 10.0f>();
  }

  SUBCASE("sqrt - F32")
  {
    run.operator()<popnet_compute_sqrt, [](float v) { return std::sqrt(v); }, POPNET_TYPE_F32, 0.0f, 100.0f>();
  }

  SUBCASE("exp - F32")
  {
    run.operator()<popnet_compute_exp, [](float v) { return std::exp(v); }, POPNET_TYPE_F32, -10.0f, 10.0f>();
  }

  SUBCASE("log - F32")
  {
    run.operator()<popnet_compute_log, [](float v) { return std::log(v); }, POPNET_TYPE_F32, 0.001f, 100.0f>();
  }

  SUBCASE("sin - F32")
  {
    run.operator()<popnet_compute_sin, [](float v) { return std::sin(v); }, POPNET_TYPE_F32, -2.0f * pi_v<float>,
                   2.0f * pi_v<float>>();
  }

  SUBCASE("cos - F32")
  {
    run.operator()<popnet_compute_cos, [](float v) { return std::cos(v); }, POPNET_TYPE_F32, -2.0f * pi_v<float>,
                   2.0f * pi_v<float>>();
  }

  SUBCASE("tanh - F32")
  {
    run.operator()<popnet_compute_tanh, [](float v) { return std::tanh(v); }, POPNET_TYPE_F32, -10.0f, 10.0f>();
  }

  SUBCASE("relu - F32")
  {
    run.operator()<popnet_compute_relu, [](float v) { return v > 0.f ? v : 0.f; }, POPNET_TYPE_F32, -10.0f, 10.0f>();
  }

  SUBCASE("sigmoid - F32")
  {
    run.operator()<popnet_compute_sigmoid, [](float v) { return 1.f / (1.f + std::exp(-v)); }, POPNET_TYPE_F32, -10.0f,
                   10.0f>();
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

template <popnet_type dtype, auto compute_ref>
static auto binary_ref(const auto& lhs, const popnet_shape& lhs_shape, const auto& rhs, const popnet_shape& rhs_shape,
                       const popnet_shape& out_shape)
{
  using traits = traits<dtype>;
  using scalar_t = typename traits::scalar_t;

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

TEST_CASE("element binary compute")
{
  auto run =
      []<auto compute, auto compute_ref, popnet_type dtype, auto lower, auto upper, bool exclude_zero = false>() {
        using traits = traits<dtype>;
        using scalar_t = typename traits::scalar_t;

        auto run_shape = [](const popnet_shape lhs_shape, const popnet_shape rhs_shape) {
          const auto dst_shape = popnet_broadcast_shape(lhs_shape, rhs_shape);

          const auto lhs_n = popnet_numel(lhs_shape);
          const auto rhs_n = popnet_numel(rhs_shape);
          const auto dst_n = popnet_numel(dst_shape);

          const auto lhs_data = generate_scalar_data<dtype>(lhs_n, lower, upper, exclude_zero, 0);
          const auto rhs_data = generate_scalar_data<dtype>(rhs_n, lower, upper, exclude_zero, 1);

          const auto expected_data =
              binary_ref<dtype, compute_ref>(lhs_data, lhs_shape, rhs_data, rhs_shape, dst_shape);

          popnet_tensor* lhs = popnet_empty(dtype, lhs_shape);
          popnet_tensor* rhs = popnet_empty(dtype, rhs_shape);
          popnet_tensor* dst = popnet_empty(dtype, dst_shape);

          REQUIRE(lhs != nullptr);
          REQUIRE(rhs != nullptr);
          REQUIRE(dst != nullptr);

          REQUIRE(popnet_set_data(lhs, lhs_data.data(), dtype, 0, lhs_n));
          REQUIRE(popnet_set_data(rhs, rhs_data.data(), dtype, 0, rhs_n));

          dst->src[0] = lhs;
          dst->src[1] = rhs;

          compute(dst);

          std::vector<scalar_t> output_data(dst_n);

          REQUIRE(popnet_get_data(dst, output_data.data(), dtype, 0, dst_n));

          for (size_t i = 0; i < dst_n; ++i)
          {
            const auto actual = output_data[i];
            const auto expected = expected_data[i];

            if constexpr (std::is_floating_point_v<decltype(expected)>)
            {
              CHECK(actual == doctest::Approx(expected));
            }
            else
            {
              CHECK(actual == expected);
            }
          }

          popnet_free(lhs);
          popnet_free(rhs);
          popnet_free(dst);
        };

        SUBCASE("1x1x1x1 1x1x1x1") { run_shape(popnet_shape{{1, 1, 1, 1}}, popnet_shape{{1, 1, 1, 1}}); }
        SUBCASE("1x1x1x2 1x1x1x2") { run_shape(popnet_shape{{1, 1, 1, 2}}, popnet_shape{{1, 1, 1, 2}}); }
        SUBCASE("1x1x1x3 1x1x1x3") { run_shape(popnet_shape{{1, 1, 1, 3}}, popnet_shape{{1, 1, 1, 3}}); }
        SUBCASE("1x1x1x4 1x1x1x4") { run_shape(popnet_shape{{1, 1, 1, 4}}, popnet_shape{{1, 1, 1, 4}}); }
        SUBCASE("1x1x1x5 1x1x1x5") { run_shape(popnet_shape{{1, 1, 1, 5}}, popnet_shape{{1, 1, 1, 5}}); }
        SUBCASE("1x1x1x1000 1x1x1x1000") { run_shape(popnet_shape{{1, 1, 1, 1000}}, popnet_shape{{1, 1, 1, 1000}}); }
        SUBCASE("1x1x1x(simd) 1x1x1x(simd)")
        {
          const size_t n = traits::vector_length();
          run_shape(popnet_shape{{1, 1, 1, n}}, popnet_shape{{1, 1, 1, n}});
        }
        SUBCASE("1x1x1x(simd+1) 1x1x1x(simd+1)")
        {
          const size_t n = traits::vector_length() + 1;
          run_shape(popnet_shape{{1, 1, 1, n}}, popnet_shape{{1, 1, 1, n}});
        }
        SUBCASE("1x1x1x(simd*16) 1x1x1x(simd*16)")
        {
          const size_t n = traits::vector_length() * 16;
          run_shape(popnet_shape{{1, 1, 1, n}}, popnet_shape{{1, 1, 1, n}});
        }
        SUBCASE("1x1x1x(simd*16+1) 1x1x1x(simd*16+1)")
        {
          const size_t n = traits::vector_length() * 16 + 1;
          run_shape(popnet_shape{{1, 1, 1, n}}, popnet_shape{{1, 1, 1, n}});
        }

        SUBCASE("1x1x1x1 1x1x1x8") { run_shape(popnet_shape{{1, 1, 1, 1}}, popnet_shape{{1, 1, 1, 8}}); }
        SUBCASE("1x1x1x16 1x1x1x1") { run_shape(popnet_shape{{1, 1, 1, 16}}, popnet_shape{{1, 1, 1, 1}}); }
      };

  SUBCASE("add - F32")
  {
    run.operator()<popnet_compute_add, [](float a, float b) { return a + b; }, POPNET_TYPE_F32, -100.0f, 100.0f>();
  }

  SUBCASE("sub - F32")
  {
    run.operator()<popnet_compute_sub, [](float a, float b) { return a - b; }, POPNET_TYPE_F32, -100.0f, 100.0f>();
  }

  SUBCASE("mul - F32")
  {
    run.operator()<popnet_compute_mul, [](float a, float b) { return a * b; }, POPNET_TYPE_F32, -10.0f, 10.0f>();
  }

  SUBCASE("div - F32")
  {
    run.operator()<popnet_compute_div, [](float a, float b) { return a / b; }, POPNET_TYPE_F32, -10.0f, 10.0f, true>();
  }
}
