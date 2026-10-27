#include <doctest/doctest.h>

#include "elemwise.h"
#include "tensor.h"

// @verifies REQ-0003: Public reduction constructors apply each valid axis mask to the output shape.
TEST_CASE("reduction output shapes follow the axis mask")
{
  const auto reductions = {popnet_reduce_sum, popnet_reduce_mean, popnet_reduce_product, popnet_reduce_min,
                           popnet_reduce_max};

  for (auto reduce : reductions)
  {
    for (const popnet_shape shape : {popnet_shape{{2, 3, 4, 5}}, popnet_shape{{1, 3, 1, 2}}})
    {
      auto* src = popnet_empty(popnet_dtype_f32, shape);

      REQUIRE(src != nullptr);

      for (uint8_t mask = 1; mask < 16; ++mask)
      {
        CAPTURE(mask);
        popnet_shape expected = shape;

        for (size_t axis = 0; axis < POPNET_MAX_DIMS; ++axis)
        {
          if (mask & (1 << axis)) expected.dim[axis] = 1;
        }

        auto* dst = reduce(src, mask);

        REQUIRE(dst != nullptr);
        CHECK(popnet_shapeof(dst) == expected);

        popnet_free(dst);
      }

      popnet_free(src);
    }
  }
}

// @verifies REQ-0002: A broadcast addition result has the independently specified shape.
// @verifies REQ-0017: Creation records the operation and inputs with independent output storage.
// @verifies REQ-0024: A result created from float32 operands has float32 dtype.
TEST_CASE("binary operation creation records result metadata")
{
  auto* a = popnet_empty(popnet_dtype_f32, {{1, 2, 1, 3}});
  auto* b = popnet_empty(popnet_dtype_f32, {{1, 1, 4, 1}});

  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);

  auto* dst = popnet_add(a, b);

  REQUIRE(dst != nullptr);
  CHECK(popnet_shapeof(dst) == popnet_shape{{1, 2, 4, 3}});
  CHECK(popnet_dtypeof(dst) == popnet_dtype_f32);
  CHECK(popnet_opof(dst) == popnet_op_add);
  CHECK(dst->src[0] == a);
  CHECK(dst->src[1] == b);

  for (size_t i = 2; i < POPNET_MAX_SRC; ++i)
  {
    CHECK(dst->src[i] == nullptr);
  }

  CHECK(dst->data != a->data);
  CHECK(dst->data != b->data);

  popnet_free(dst);
  popnet_free(b);
  popnet_free(a);
}

// @verifies REQ-0022: Data transfers and repeated computation preserve tensor metadata and dependencies.
TEST_CASE("operation metadata stays fixed during computation")
{
  const popnet_shape shape{{1, 1, 1, 4}};
  auto* a = popnet_empty(popnet_dtype_f32, shape);
  auto* b = popnet_empty(popnet_dtype_f32, shape);

  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);

  auto* dst = popnet_add(a, b);

  REQUIRE(dst != nullptr);

  const auto inputs = dst->src;

  for (int pass = 0; pass < 2; ++pass)
  {
    const float values[] = {float(pass), 1, 2, 3};
    popnet_set_data(a, values, popnet_dtype_f32, 0, 4);
    popnet_set_data(b, values, popnet_dtype_f32, 0, 4);

    popnet_compute_add(dst);

    float output[4]{};

    popnet_get_data(dst, output, popnet_dtype_f32, 0, 4);

    CHECK(popnet_shapeof(dst) == shape);
    CHECK(popnet_dtypeof(dst) == popnet_dtype_f32);
    CHECK(popnet_opof(dst) == popnet_op_add);
    CHECK(dst->src == inputs);

    for (auto* src : {a, b})
    {
      CHECK(popnet_shapeof(src) == shape);
      CHECK(popnet_dtypeof(src) == popnet_dtype_f32);
      CHECK(popnet_opof(src) == popnet_op_none);

      for (auto* dependency : src->src)
      {
        CHECK(dependency == nullptr);
      }
    }
  }

  popnet_free(dst);
  popnet_free(b);
  popnet_free(a);
}
