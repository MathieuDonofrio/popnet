#include <cstring>
#include <doctest/doctest.h>
#include <limits>
#include <vector>

#include "elemwise.h"
#include "linalg.h"
#include "reduce.h"
#include "tensor.h"

// @verifies REQ-0025: Direct computation leaves all operand buffers unchanged.
TEST_CASE("compute operations preserve input data")
{
  auto check = [](auto compute, popnet_shape shape, popnet_shape output_shape, bool binary = false,
                  popnet_shape rhs_shape = {{1, 1, 1, 1}}, bool special_values = false) {
    auto* src = popnet_empty(popnet_dtype_f32, shape);
    auto* dst = popnet_empty(popnet_dtype_f32, output_shape);

    REQUIRE(src != nullptr);
    REQUIRE(dst != nullptr);

    std::vector<float> values(popnet_numel(shape));

    for (size_t i = 0; i < values.size(); ++i)
    {
      values[i] = 0.5f + float(i % 7);
    }

    if (special_values)
    {
      values[0] = -0.0f;
      values[1] = std::numeric_limits<float>::quiet_NaN();
      values[2] = std::numeric_limits<float>::infinity();
      values[3] = -std::numeric_limits<float>::infinity();
    }

    popnet_set_data(src, values.data(), popnet_dtype_f32, 0, values.size());
    dst->src[0] = src;

    popnet_tensor* rhs = nullptr;
    std::vector<float> rhs_values;
    if (binary)
    {
      rhs = popnet_empty(popnet_dtype_f32, rhs_shape);

      REQUIRE(rhs != nullptr);

      rhs_values.resize(popnet_numel(rhs_shape));

      for (size_t i = 0; i < rhs_values.size(); ++i)
      {
        rhs_values[i] = 1.25f + float(i % 5);
      }

      popnet_set_data(rhs, rhs_values.data(), popnet_dtype_f32, 0, rhs_values.size());
      dst->src[1] = rhs;
    }

    compute(dst);

    std::vector<float> actual(values.size());

    popnet_get_data(src, actual.data(), popnet_dtype_f32, 0, actual.size());

    CHECK(std::memcmp(actual.data(), values.data(), values.size() * sizeof(float)) == 0);
    if (binary)
    {
      actual.resize(rhs_values.size());
      popnet_get_data(rhs, actual.data(), popnet_dtype_f32, 0, actual.size());

      CHECK(std::memcmp(actual.data(), rhs_values.data(), rhs_values.size() * sizeof(float)) == 0);
    }

    popnet_free(dst);
    popnet_free(rhs);
    popnet_free(src);
  };

  struct operation
  {
    const char* name;
    void (*compute)(popnet_tensor*);
  };

  SUBCASE("unary")
  {
    const operation operations[] = {
        {"abs", popnet_compute_abs},   {"neg", popnet_compute_neg},        {"sqr", popnet_compute_sqr},
        {"sqrt", popnet_compute_sqrt}, {"exp", popnet_compute_exp},        {"log", popnet_compute_log},
        {"sin", popnet_compute_sin},   {"cos", popnet_compute_cos},        {"tanh", popnet_compute_tanh},
        {"relu", popnet_compute_relu}, {"sigmoid", popnet_compute_sigmoid}};

    for (auto op : operations)
    {
      CAPTURE(op.name);
      check(op.compute, {{1, 1, 1, 33}}, {{1, 1, 1, 33}});
    }
  }

  SUBCASE("binary")
  {
    const operation operations[] = {{"add", popnet_compute_add},
                                    {"sub", popnet_compute_sub},
                                    {"mul", popnet_compute_mul},
                                    {"div", popnet_compute_div}};

    for (auto op : operations)
    {
      CAPTURE(op.name);
      check(op.compute, {{1, 1, 1, 33}}, {{1, 1, 1, 33}}, true, {{1, 1, 1, 33}});
      check(op.compute, {{1, 1, 1, 1}}, {{1, 1, 1, 33}}, true, {{1, 1, 1, 33}});
      check(op.compute, {{1, 1, 1, 33}}, {{1, 1, 1, 33}}, true);
      check(op.compute, {{1, 2, 1, 3}}, {{1, 2, 4, 3}}, true, {{1, 1, 4, 1}});
    }
  }

  SUBCASE("reductions")
  {
    const operation operations[] = {{"sum", popnet_compute_reduce_sum},
                                    {"mean", popnet_compute_reduce_mean},
                                    {"product", popnet_compute_reduce_product},
                                    {"min", popnet_compute_reduce_min},
                                    {"max", popnet_compute_reduce_max}};

    for (auto op : operations)
    {
      CAPTURE(op.name);
      check(op.compute, {{1, 1, 1, 1}}, {{1, 1, 1, 1}});
      check(op.compute, {{1, 1, 2, 33}}, {{1, 1, 1, 1}});
      check(op.compute, {{1, 1, 2, 33}}, {{1, 1, 2, 1}});
      check(op.compute, {{2, 3, 1, 1}}, {{2, 3, 1, 1}}, false, {}, true);
      check(op.compute, {{1, 3, 1, 2}}, {{1, 3, 1, 2}}, false, {}, true);
    }
  }

  SUBCASE("matmul") { check(popnet_compute_matmul, {{1, 1, 2, 3}}, {{1, 1, 2, 4}}, true, {{1, 1, 4, 3}}); }

  SUBCASE("transpose") { check(popnet_compute_transpose, {{1, 1, 2, 3}}, {{1, 1, 3, 2}}); }
}
