#include <doctest/doctest.h>

#include "elemwise.h"
#include "tensor.h"

// @verifies REQ-0003: Public reduction constructors apply each valid axis mask to the output shape.
// @verifies REQ-0017: Reduction results record their operation and input with independent storage.
// @verifies REQ-0024: Reduction results preserve float32 dtype.
TEST_CASE("reduction output shapes follow the axis mask")
{
  const struct
  {
    decltype(&popnet_reduce_sum) create;
    popnet_op op;
  } reductions[] = {{popnet_reduce_sum, popnet_op_reduce_sum},
                    {popnet_reduce_mean, popnet_op_reduce_mean},
                    {popnet_reduce_product, popnet_op_reduce_product},
                    {popnet_reduce_min, popnet_op_reduce_min},
                    {popnet_reduce_max, popnet_op_reduce_max}};

  for (auto reduce : reductions)
  {
    CAPTURE(reduce.op);
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

        auto* dst = reduce.create(src, mask);

        REQUIRE(dst != nullptr);
        CHECK(popnet_shapeof(dst) == expected);
        CHECK(popnet_dtypeof(dst) == popnet_dtype_f32);
        CHECK(popnet_opof(dst) == reduce.op);
        CHECK(dst->src[0] == src);
        CHECK(dst->data != src->data);

        for (size_t i = 1; i < POPNET_MAX_SRC; ++i)
        {
          CHECK(dst->src[i] == nullptr);
        }

        popnet_free(dst);
      }

      popnet_free(src);
    }
  }
}

// @verifies REQ-0002: Each binary constructor applies the same broadcasting contract.
// @verifies REQ-0017: Creation records the operation and inputs with independent output storage.
// @verifies REQ-0024: A result created from float32 operands has float32 dtype.
TEST_CASE("binary operation creation records result metadata")
{
  auto* a = popnet_empty(popnet_dtype_f32, {{1, 2, 1, 3}});
  auto* b = popnet_empty(popnet_dtype_f32, {{1, 1, 4, 1}});

  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);

  const struct
  {
    decltype(&popnet_add) create;
    popnet_op op;
  } operations[] = {{popnet_add, popnet_op_add},
                    {popnet_sub, popnet_op_sub},
                    {popnet_mul, popnet_op_mul},
                    {popnet_div, popnet_op_div}};

  for (auto operation : operations)
  {
    CAPTURE(operation.op);

    auto* dst = operation.create(a, b);

    REQUIRE(dst != nullptr);
    CHECK(popnet_shapeof(dst) == popnet_shape{{1, 2, 4, 3}});
    CHECK(popnet_dtypeof(dst) == popnet_dtype_f32);
    CHECK(popnet_opof(dst) == operation.op);
    CHECK(dst->src[0] == a);
    CHECK(dst->src[1] == b);

    for (size_t i = 2; i < POPNET_MAX_SRC; ++i)
    {
      CHECK(dst->src[i] == nullptr);
    }

    CHECK(dst->data != a->data);
    CHECK(dst->data != b->data);

    popnet_free(dst);
  }

  popnet_free(b);
  popnet_free(a);
}

// @verifies REQ-0001: Unary constructors preserve the input shape.
// @verifies REQ-0017: Unary results record their operation and input with independent storage.
// @verifies REQ-0024: Unary results preserve float32 dtype.
TEST_CASE("unary operation creation records result metadata")
{
  const popnet_shape shape{{2, 3, 4, 5}};
  auto* src = popnet_empty(popnet_dtype_f32, shape);

  REQUIRE(src != nullptr);

  const struct
  {
    decltype(&popnet_abs) create;
    popnet_op op;
  } operations[] = {{popnet_abs, popnet_op_abs},   {popnet_neg, popnet_op_neg},        {popnet_sqr, popnet_op_sqr},
                    {popnet_sqrt, popnet_op_sqrt}, {popnet_exp, popnet_op_exp},        {popnet_log, popnet_op_log},
                    {popnet_sin, popnet_op_sin},   {popnet_cos, popnet_op_cos},        {popnet_tanh, popnet_op_tanh},
                    {popnet_relu, popnet_op_relu}, {popnet_sigmoid, popnet_op_sigmoid}};

  for (auto operation : operations)
  {
    CAPTURE(operation.op);

    auto* dst = operation.create(src);

    REQUIRE(dst != nullptr);
    CHECK(popnet_shapeof(dst) == shape);
    CHECK(popnet_dtypeof(dst) == popnet_dtype_f32);
    CHECK(popnet_opof(dst) == operation.op);
    CHECK(dst->src[0] == src);
    CHECK(dst->data != src->data);

    for (size_t i = 1; i < POPNET_MAX_SRC; ++i)
    {
      CHECK(dst->src[i] == nullptr);
    }

    popnet_free(dst);
  }

  popnet_free(src);
}

// @verifies REQ-0004: Matrix multiplication preserves outer dimensions and uses the right operand's row count.
// @verifies REQ-0017: Matrix multiplication records both inputs with independent result storage.
// @verifies REQ-0024: Matrix multiplication results preserve float32 dtype.
TEST_CASE("matrix multiplication creation records result metadata")
{
  auto* a = popnet_empty(popnet_dtype_f32, {{2, 3, 4, 5}});
  auto* b = popnet_empty(popnet_dtype_f32, {{1, 1, 6, 5}});

  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);

  auto* dst = popnet_matmul(a, b);

  REQUIRE(dst != nullptr);
  CHECK(popnet_shapeof(dst) == popnet_shape{{2, 3, 4, 6}});
  CHECK(popnet_dtypeof(dst) == popnet_dtype_f32);
  CHECK(popnet_opof(dst) == popnet_op_matmul);
  CHECK(dst->src[0] == a);
  CHECK(dst->src[1] == b);
  CHECK(dst->data != a->data);
  CHECK(dst->data != b->data);

  for (size_t i = 2; i < POPNET_MAX_SRC; ++i)
  {
    CHECK(dst->src[i] == nullptr);
  }

  popnet_free(dst);
  popnet_free(b);
  popnet_free(a);
}

// @verifies REQ-0005: Transpose construction swaps only the final two dimensions.
// @verifies REQ-0017: Transpose records its input with independent result storage.
// @verifies REQ-0024: Transpose results preserve float32 dtype.
TEST_CASE("transpose creation records result metadata")
{
  auto* src = popnet_empty(popnet_dtype_f32, {{2, 3, 4, 5}});

  REQUIRE(src != nullptr);

  auto* dst = popnet_transpose(src);

  REQUIRE(dst != nullptr);
  CHECK(popnet_shapeof(dst) == popnet_shape{{2, 3, 5, 4}});
  CHECK(popnet_dtypeof(dst) == popnet_dtype_f32);
  CHECK(popnet_opof(dst) == popnet_op_transpose);
  CHECK(dst->src[0] == src);
  CHECK(dst->data != src->data);

  for (size_t i = 1; i < POPNET_MAX_SRC; ++i)
  {
    CHECK(dst->src[i] == nullptr);
  }

  popnet_free(dst);
  popnet_free(src);
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
