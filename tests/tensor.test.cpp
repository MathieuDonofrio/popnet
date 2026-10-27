#include "tensor.h"

#include <doctest/doctest.h>

#include "dtype_traits.h"

TEST_CASE("popnet_numel computes number of elements correctly")
{
  popnet_shape shape{{2, 3, 4, 5}};
  REQUIRE(popnet_numel(shape) == 2 * 3 * 4 * 5);
}

TEST_CASE("popnet_empty and popnet_empty_like")
{
  auto run = []<popnet_dtype dtype>() {
    popnet_shape shape{{2, 2, 2, 2}};
    popnet_tensor* t = popnet_empty(dtype, shape);

    REQUIRE(t != nullptr);
    CHECK(popnet_valid_tensor(t));
    CHECK(popnet_dtypeof(t) == dtype);
    CHECK(popnet_opof(t) == popnet_op_none);
    CHECK(popnet_shapeof(t) == shape);

    t->op = popnet_op_abs;

    SUBCASE("empty_like preserves shape and dtype but not op")
    {
      popnet_tensor* t2 = popnet_empty_like(t);

      REQUIRE(t2 != nullptr);
      CHECK(popnet_valid_tensor(t2));
      CHECK(popnet_dtypeof(t2) == dtype);
      CHECK(popnet_opof(t2) == popnet_op_none);
      CHECK(popnet_shapeof(t2) == shape);
      CHECK(t2->data != t->data);

      popnet_free(t2);
    }

    SUBCASE("empty_like with nullptr should fail gracefully")
    {
      popnet_tensor* bad = popnet_empty_like(nullptr);

      REQUIRE(bad == nullptr);
      CHECK_FALSE(popnet_valid_tensor(bad));
      CHECK(popnet_dtypeof(bad) == popnet_dtype_none);
      CHECK(popnet_opof(bad) == popnet_op_none);
      CHECK(popnet_shapeof(bad) == popnet_shape{{0, 0, 0, 0}});
    }

    popnet_free(t);
  };

  SUBCASE("F32") { run.operator()<popnet_dtype_f32>(); }
}

TEST_CASE("popnet_set_data and popnet_get_data")
{
  auto run = []<popnet_dtype dtype>() {
    popnet_shape shape{{2, 2, 1, 1}};
    popnet_tensor* t = popnet_empty(dtype, shape);
    REQUIRE(t != nullptr);

    typename dtype_traits<dtype>::scalar_t values[4] = {0, 1, 0, -1};
    typename dtype_traits<dtype>::scalar_t readback[4] = {};

    SUBCASE("Full write and readback")
    {
      popnet_set_data(t, values, dtype, 0, 4);
      popnet_get_data(t, readback, dtype, 0, 4);

      for (int i = 0; i < 4; i++)
      {
        REQUIRE(readback[i] == values[i]);
      }
    }

    SUBCASE("Partial write and readback with offset")
    {
      popnet_set_data(t, values, dtype, 1, 2);
      popnet_get_data(t, readback, dtype, 1, 2);

      REQUIRE(readback[0] == values[0]);
      REQUIRE(readback[1] == values[1]);
    }

    popnet_free(t);
  };

  SUBCASE("F32") { run.operator()<popnet_dtype_f32>(); }
}
