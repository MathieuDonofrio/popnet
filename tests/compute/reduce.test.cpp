#include "reduce.h"

#include <doctest/doctest.h>

#include "simd.h"
#include "tensor.h"
#include "test_utils.h"

TEST_CASE("full reduce compute")
{
  auto run =
      []<auto compute, auto compute_ref, popnet_type dtype, auto lower, auto upper, bool exclude_zero = false>() {
        auto run_shape = [&](size_t d0, size_t d1, size_t d2, size_t d3) {
          popnet_shape shape{{d0, d1, d2, d3}};

          size_t n = popnet_nelems(shape);

          auto input_data = generate_scalar_data<dtype>(n, lower, upper, exclude_zero);

          auto expected = compute_ref(input_data);

          popnet_tensor* src = popnet_empty(dtype, shape);
          popnet_tensor* dst = popnet_empty(dtype, {{1, 1, 1, 1}});

          REQUIRE(src != nullptr);
          REQUIRE(dst != nullptr);

          REQUIRE(popnet_set_data(src, input_data.data(), dtype, 0, input_data.size()));

          dst->src[0] = src;

          compute(dst);

          decltype(expected) actual{};
          REQUIRE(popnet_get_data(dst, &actual, dtype, 0, 1));

          if constexpr (std::is_floating_point_v<decltype(expected)>)
          {
            CHECK(actual == doctest::Approx(expected));
          }
          else
          {
            CHECK(actual == expected);
          }

          popnet_free(src);
          popnet_free(dst);
        };

        SUBCASE("1x1x1x1") { run_shape(1, 1, 1, 1); }
        SUBCASE("2x2x2x2") { run_shape(2, 2, 2, 2); }
        SUBCASE("2x3x4x5") { run_shape(2, 3, 4, 5); }
        SUBCASE("4x4x4x4") { run_shape(4, 4, 4, 4); }
        SUBCASE("8x2x3x5") { run_shape(8, 2, 3, 5); }

        SUBCASE("1x1x1x(simd)") { run_shape(1, 1, 1, simd_traits<dtype>::length()); }
        SUBCASE("1x1x1x(simd+1)") { run_shape(1, 1, 1, simd_traits<dtype>::length() + 1); }
        SUBCASE("1x1x1x(simd*4)") { run_shape(1, 1, 1, simd_traits<dtype>::length() * 4); }
        SUBCASE("1x1x1x(simd*4+1)") { run_shape(1, 1, 1, simd_traits<dtype>::length() * 4 + 1); }
        SUBCASE("(simd)x1x1x1") { run_shape(simd_traits<dtype>::length(), 1, 1, 1); }
        SUBCASE("(simd+1)x1x1x1") { run_shape(simd_traits<dtype>::length() + 1, 1, 1, 1); }
        SUBCASE("(simd*4)x1x1x1") { run_shape(simd_traits<dtype>::length() * 4, 1, 1, 1); }
        SUBCASE("(simd*4+1)x1x1x1") { run_shape(simd_traits<dtype>::length() * 4 + 1, 1, 1, 1); }
        SUBCASE("1x(simd)x1x1") { run_shape(1, simd_traits<dtype>::length(), 1, 1); }
        SUBCASE("1x(simd+1)x1x1") { run_shape(1, simd_traits<dtype>::length() + 1, 1, 1); }
        SUBCASE("1x(simd*4)x1x1") { run_shape(1, simd_traits<dtype>::length() * 4, 1, 1); }
        SUBCASE("1x(simd*4+1)x1x1") { run_shape(1, simd_traits<dtype>::length() * 4 + 1, 1, 1); }
        SUBCASE("1x1x(simd)x1") { run_shape(1, 1, simd_traits<dtype>::length(), 1); }
        SUBCASE("1x1x(simd+1)x1") { run_shape(1, 1, simd_traits<dtype>::length() + 1, 1); }
        SUBCASE("1x1x(simd*4)x1") { run_shape(1, 1, simd_traits<dtype>::length() * 4, 1); }
        SUBCASE("1x1x(simd*4+1)x1") { run_shape(1, 1, simd_traits<dtype>::length() * 4 + 1, 1); }

        SUBCASE("(simd+1)x2x2x2") { run_shape(simd_traits<dtype>::length() + 1, 2, 2, 2); }
        SUBCASE("(simd)x(simd)x2x2") { run_shape(simd_traits<dtype>::length(), simd_traits<dtype>::length(), 2, 2); }
        SUBCASE("(simd)x1x(simd)x1") { run_shape(simd_traits<dtype>::length(), 1, simd_traits<dtype>::length(), 1); }
        SUBCASE("1x(simd)x1x(simd)") { run_shape(1, simd_traits<dtype>::length(), 1, simd_traits<dtype>::length()); }
      };

  SUBCASE("sum - F32")
  {
    run.operator()<popnet_compute_reduce_sum,
                   [](const std::vector<float>& v) { return std::accumulate(v.begin(), v.end(), 0.0f); },
                   POPNET_TYPE_F32, -10.0f, 10.0f>();
  }

  SUBCASE("mean - F32")
  {
    run.operator()<popnet_compute_reduce_mean,
                   [](const std::vector<float>& v) {
                     return std::accumulate(v.begin(), v.end(), 0.0f) / static_cast<float>(v.size());
                   },
                   POPNET_TYPE_F32, -10.0f, 10.0f>();
  }

  SUBCASE("product - F32")
  {
    run.operator()<popnet_compute_reduce_product,
                   [](const std::vector<float>& v) {
                     return std::accumulate(v.begin(), v.end(), 1.0f, std::multiplies<float>());
                   },
                   POPNET_TYPE_F32, 0.5f, 2.0f>(); // avoids zero underflow
  }

  SUBCASE("min - F32")
  {
    run.operator()<popnet_compute_reduce_min,
                   [](const std::vector<float>& v) { return *std::min_element(v.begin(), v.end()); }, POPNET_TYPE_F32,
                   -100.0f, 100.0f>();
  }

  SUBCASE("max - F32")
  {
    run.operator()<popnet_compute_reduce_max,
                   [](const std::vector<float>& v) { return *std::max_element(v.begin(), v.end()); }, POPNET_TYPE_F32,
                   -100.0f, 100.0f>();
  }
}
