#include <doctest/doctest.h>

#include "simd.h"
#include "tensor.h"

TEST_CASE("popnet_nelems computes number of elements correctly")
{
  popnet_shape shape{{2, 3, 4, 5}};
  REQUIRE(popnet_nelems(shape) == 2 * 3 * 4 * 5);
}

TEST_CASE("popnet_empty and popnet_empty_like")
{
  auto run_test = []<popnet_type DType>() {
    popnet_shape shape{{2, 2, 2, 2}};
    popnet_tensor* t = popnet_empty(POPNET_TYPE_F32, shape);

    REQUIRE(t != nullptr);
    REQUIRE(popnet_valid_tensor(t) == true);
    REQUIRE(popnet_nelems(popnet_shapeof(t)) == 16);
    REQUIRE(popnet_dtypeof(t) == POPNET_TYPE_F32);

    SUBCASE("empty_like preserves shape and dtype")
    {
      popnet_tensor* t2 = popnet_empty_like(t);
      REQUIRE(t2 != nullptr);
      REQUIRE(popnet_valid_tensor(t2) == true);
      REQUIRE(popnet_dtypeof(t2) == POPNET_TYPE_F32);

      auto shape2 = popnet_shapeof(t2);
      for (size_t i = 0; i < POPNET_MAX_DIMS; ++i)
      {
        REQUIRE(shape2.dim[i] == shape.dim[i]);
      }

      REQUIRE(t2->data != t->data);

      popnet_free(t2);
    }

    SUBCASE("empty_like with nullptr should fail gracefully")
    {
      popnet_tensor* bad = popnet_empty_like(nullptr);
      REQUIRE(bad == nullptr);
    }

    popnet_free(t);
  };

  SUBCASE("dtype = F32") { run_test.operator()<POPNET_TYPE_F32>(); }
}

TEST_CASE("popnet_set_data and popnet_get_data")
{
  auto run_for_dtype = []<popnet_type DType>() {
    popnet_shape shape{{2, 2, 1, 1}};
    popnet_tensor* t = popnet_empty(POPNET_TYPE_F32, shape);
    REQUIRE(t != nullptr);

    typename simd<DType>::scalar_t values[4] = {0, 1, 0, -1};
    typename simd<DType>::scalar_t readback[4] = {};

    SUBCASE("Full write and readback")
    {
      REQUIRE(popnet_set_data(t, values, POPNET_TYPE_F32, 0, 4) == true);
      REQUIRE(popnet_get_data(t, readback, POPNET_TYPE_F32, 0, 4) == true);

      for (int i = 0; i < 4; i++)
      {
        REQUIRE(readback[i] == values[i]);
      }
    }

    SUBCASE("Partial write and readback with offset")
    {
      REQUIRE(popnet_set_data(t, values, POPNET_TYPE_F32, 1, 2) == true);
      REQUIRE(popnet_get_data(t, readback, POPNET_TYPE_F32, 1, 2) == true);

      REQUIRE(readback[0] == values[0]);
      REQUIRE(readback[1] == values[1]);
    }

    popnet_free(t);
  };

  SUBCASE("F32") { run_for_dtype.operator()<POPNET_TYPE_F32>(); }
}
