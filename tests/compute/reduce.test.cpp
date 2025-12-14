#include "reduce.h"

#include <doctest/doctest.h>
#include <limits>
#include <vector>

#include "tensor.h"
#include "test_utils.h"
#include "traits.h"

template <popnet_type dtype, class InitFn, class MergeFn, class FinalizeFn>
static auto reduce_ref(const auto& src, popnet_shape src_shape, auto& dst, popnet_shape dst_shape, InitFn init_fn,
                       MergeFn merge_fn, FinalizeFn finalize_fn)
{
  using traits = traits<dtype>;
  using scalar_t = typename traits::scalar_t;

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

TEST_CASE("reduce compute")
{
  auto run =
      []<auto compute, auto compute_ref, popnet_type dtype, auto lower, auto upper, bool exclude_zero = false>() {
        auto run_shape = [&](popnet_shape src_shape, popnet_shape dst_shape) {
          using dtype_traits = traits<dtype>;
          using scalar_t = typename dtype_traits::scalar_t;

          const size_t n = popnet_numel(src_shape);

          auto input_data = generate_scalar_data<dtype>(n, lower, upper, exclude_zero);

          auto expected = compute_ref(input_data, src_shape, dst_shape);

          popnet_tensor* src = popnet_empty(dtype, src_shape);
          popnet_tensor* dst = popnet_empty(dtype, dst_shape);

          REQUIRE(src != nullptr);
          REQUIRE(dst != nullptr);

          REQUIRE(popnet_set_data(src, input_data.data(), dtype, 0, input_data.size()));

          dst->src[0] = src;

          compute(dst);

          std::vector<scalar_t> actual(popnet_numel(dst_shape));
          REQUIRE(popnet_get_data(dst, actual.data(), dtype, 0, actual.size()));

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
        SUBCASE("2x2x2x2 -> 1x1x1x1") { run_shape({{2, 2, 2, 2}}, popnet_scalar_shape); }
        SUBCASE("2x3x4x5 -> 1x1x1x1") { run_shape({{2, 3, 4, 5}}, popnet_scalar_shape); }
        SUBCASE("4x4x4x4 -> 1x1x1x1") { run_shape({{4, 4, 4, 4}}, popnet_scalar_shape); }
        SUBCASE("8x2x3x5 -> 1x1x1x1") { run_shape({{8, 2, 3, 5}}, popnet_scalar_shape); }
        SUBCASE("3x4x5x1 -> 1x1x1x1") { run_shape({{3, 4, 5, 1}}, popnet_scalar_shape); }
        SUBCASE("3x4x1x1 -> 1x1x1x1") { run_shape({{3, 4, 1, 1}}, popnet_scalar_shape); }
        SUBCASE("3x1x1x1 -> 1x1x1x1") { run_shape({{3, 1, 1, 1}}, popnet_scalar_shape); }
        SUBCASE("12x34x56x78 -> 1x1x1x1") { run_shape({{4, 4, 4, 4}}, popnet_scalar_shape); }
        SUBCASE("2x3x4x5 -> 2x3x4x1") { run_shape({{2, 3, 4, 5}}, {{2, 3, 4, 1}}); }
        SUBCASE("2x3x4x5 -> 2x3x1x1") { run_shape({{2, 3, 4, 5}}, {{2, 3, 1, 1}}); }
        SUBCASE("2x3x4x5 -> 2x1x1x1") { run_shape({{2, 3, 4, 5}}, {{2, 1, 1, 1}}); }
      };

  SUBCASE("sum - F32")
  {
    run.operator()<popnet_compute_reduce_sum,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_type_f32>(
                         src, src_shape, dst, dst_shape, [] { return 0.0f; }, [](float& acc, float v) { acc += v; },
                         [](float acc, size_t) { return acc; });
                   },
                   popnet_type_f32, -10.0f, 10.0f>();
  }

  SUBCASE("mean - F32")
  {
    run.operator()<popnet_compute_reduce_mean,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_type_f32>(
                         src, src_shape, dst, dst_shape, [] { return 0.0f; }, [](float& acc, float v) { acc += v; },
                         [](float acc, size_t count) { return acc / count; });
                   },
                   popnet_type_f32, -10.0f, 10.0f>();
  }

  SUBCASE("product - F32")
  {
    run.operator()<popnet_compute_reduce_product,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_type_f32>(
                         src, src_shape, dst, dst_shape, [] { return 1.0f; }, [](float& acc, float v) { acc *= v; },
                         [](float acc, size_t) { return acc; });
                   },
                   popnet_type_f32, 0.5f, 2.0f>(); // avoids zero underflow
  }

  SUBCASE("min - F32")
  {
    run.operator()<popnet_compute_reduce_min,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_type_f32>(
                         src, src_shape, dst, dst_shape, [] { return std::numeric_limits<float>::infinity(); },
                         [](float& acc, float v) { acc = std::min(acc, v); }, [](float acc, size_t) { return acc; });
                   },
                   popnet_type_f32, -100.0f, 100.0f>();
  }

  SUBCASE("max - F32")
  {
    run.operator()<popnet_compute_reduce_max,
                   [](const std::vector<float>& src, popnet_shape src_shape, popnet_shape dst_shape) {
                     std::vector<float> dst;
                     return reduce_ref<popnet_type_f32>(
                         src, src_shape, dst, dst_shape, [] { return -std::numeric_limits<float>::infinity(); },
                         [](float& acc, float v) { acc = std::max(acc, v); }, [](float acc, size_t) { return acc; });
                   },
                   popnet_type_f32, -100.0f, 100.0f>();
  }
}
