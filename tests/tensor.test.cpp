#include "tensor.h"

#include <algorithm>
#include <doctest/doctest.h>

// @verifies REQ-0013: Checks that the element count is the product of the dimensions.
TEST_CASE("popnet_numel computes number of elements correctly")
{
  popnet_shape shape{{2, 3, 4, 5}};

  REQUIRE(popnet_numel(shape) == 2 * 3 * 4 * 5);
}

// @verifies REQ-0010: Checks allocation metadata and independent storage.
// @verifies REQ-0013: Checks metadata queries on newly allocated tensors.
TEST_CASE("popnet_empty")
{
  const popnet_shape shape{{2, 3, 1, 1}};
  auto* a = popnet_empty(popnet_dtype_f32, shape);
  auto* b = popnet_empty(popnet_dtype_f32, shape);

  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);
  CHECK(popnet_valid_tensor(a));
  CHECK(popnet_shapeof(a) == shape);
  CHECK(popnet_dtypeof(a) == popnet_dtype_f32);
  CHECK(popnet_opof(a) == popnet_op_none);
  CHECK(a->data != b->data);

  popnet_free(b);
  popnet_free(a);
}

// @verifies REQ-0011: Checks matching metadata, independent storage, and no copied operation.
// @verifies REQ-0013: Checks metadata queries on the matching allocation.
TEST_CASE("popnet_empty_like")
{
  const popnet_shape shape{{2, 3, 1, 1}};
  auto* input = popnet_empty(popnet_dtype_f32, shape);

  REQUIRE(input != nullptr);

  auto* src = popnet_abs(input);

  REQUIRE(src != nullptr);

  SUBCASE("matching allocation")
  {
    auto* dst = popnet_empty_like(src);

    REQUIRE(dst != nullptr);
    CHECK(popnet_valid_tensor(dst));
    CHECK(popnet_shapeof(dst) == shape);
    CHECK(popnet_dtypeof(dst) == popnet_dtype_f32);
    CHECK(popnet_opof(dst) == popnet_op_none);
    CHECK(dst->data != src->data);

    popnet_free(dst);
  }

  SUBCASE("nullptr")
  {
    auto* bad = popnet_empty_like(nullptr);
    CHECK(bad == nullptr);
    CHECK_FALSE(popnet_valid_tensor(bad));
    CHECK(popnet_dtypeof(bad) == popnet_dtype_none);
    CHECK(popnet_opof(bad) == popnet_op_none);
    CHECK(popnet_shapeof(bad) == popnet_shape{{0, 0, 0, 0}});

    popnet_free(bad);
  }

  popnet_free(src);
  popnet_free(input);
}

// @verifies REQ-0015: Checks writes and preservation outside the requested range.
TEST_CASE("popnet_set_data")
{
  auto* tensor = popnet_empty(popnet_dtype_f32, {{1, 1, 1, 4}});

  REQUIRE(tensor != nullptr);

  auto* data = static_cast<float*>(tensor->data);
  const float initial[] = {11, 22, 33, 44};
  std::copy_n(initial, 4, data);
  const float values[] = {7, -3, 9, 5};

  SUBCASE("full write")
  {
    popnet_set_data(tensor, values, popnet_dtype_f32, 0, 4);

    for (size_t i = 0; i < 4; ++i)
    {
      CHECK(data[i] == values[i]);
    }
  }

  SUBCASE("partial write")
  {
    popnet_set_data(tensor, values, popnet_dtype_f32, 1, 2);
    const float expected[] = {11, 7, -3, 44};

    for (size_t i = 0; i < 4; ++i)
    {
      CHECK(data[i] == expected[i]);
    }
  }

  SUBCASE("zero-length write at end")
  {
    popnet_set_data(tensor, values, popnet_dtype_f32, 4, 0);

    for (size_t i = 0; i < 4; ++i)
    {
      CHECK(data[i] == initial[i]);
    }
  }

  popnet_free(tensor);
}

// @verifies REQ-0016: Checks read offsets, count, and preservation of tensor data.
TEST_CASE("popnet_get_data")
{
  auto* tensor = popnet_empty(popnet_dtype_f32, {{1, 1, 1, 4}});

  REQUIRE(tensor != nullptr);

  auto* data = static_cast<float*>(tensor->data);
  const float values[] = {11, 22, 33, 44};
  std::copy_n(values, 4, data);
  float output[] = {-1, -1, -1, -1};

  SUBCASE("full read")
  {
    popnet_get_data(tensor, output, popnet_dtype_f32, 0, 4);

    for (size_t i = 0; i < 4; ++i)
    {
      CHECK(output[i] == values[i]);
    }
  }

  SUBCASE("partial read")
  {
    popnet_get_data(tensor, output, popnet_dtype_f32, 1, 2);
    const float expected[] = {22, 33, -1, -1};

    for (size_t i = 0; i < 4; ++i)
    {
      CHECK(output[i] == expected[i]);
    }
  }

  SUBCASE("zero-length read at end")
  {
    popnet_get_data(tensor, output, popnet_dtype_f32, 4, 0);

    for (float value : output)
    {
      CHECK(value == -1);
    }
  }

  for (size_t i = 0; i < 4; ++i)
  {
    CHECK(data[i] == values[i]);
  }

  popnet_free(tensor);
}

// @verifies REQ-0012: Freeing an operation tensor leaves its input alive and usable.
TEST_CASE("freeing a tensor preserves its inputs")
{
  auto* src = popnet_empty(popnet_dtype_f32, {{1, 1, 1, 2}});

  REQUIRE(src != nullptr);

  const float values[] = {3, -7};
  popnet_set_data(src, values, popnet_dtype_f32, 0, 2);
  auto* dst = popnet_neg(src);

  REQUIRE(dst != nullptr);

  popnet_free(dst);

  float actual[2]{};

  popnet_get_data(src, actual, popnet_dtype_f32, 0, 2);

  CHECK(actual[0] == values[0]);
  CHECK(actual[1] == values[1]);

  popnet_free(src);
}
