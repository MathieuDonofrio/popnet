#include "graph.h"

#include <algorithm>
#include <cmath>
#include <doctest/doctest.h>
#include <vector>

#include "ops.h"
#include "tensor.h"
#include "test_assertions.h"

template <typename T>
static void check_graph_result(popnet_tensor* result, const T* expected, size_t count, double atol = 0, double rtol = 0)
{
  REQUIRE(result != nullptr);
  REQUIRE(popnet_numel(popnet_shapeof(result)) == count);

  auto* graph = popnet_graph_alloc();

  REQUIRE(graph != nullptr);
  REQUIRE(popnet_graph_build_forward(graph, result));

  popnet_graph_compute_forward(graph);

  std::vector<float> actual(count);
  popnet_get_data(result, actual.data(), popnet_dtype_f32, 0, count);

  for (size_t i = 0; i < count; ++i)
  {
    CAPTURE(i);
    CHECK_F32_CLOSE(actual[i], expected[i], atol, rtol);
  }

  popnet_graph_free(graph);
}

// @verifies REQ-0020: Forward execution dispatches every recorded unary operation.
// @verifies REQ-0026: Forward results satisfy the operation-specific accuracy contract.
TEST_CASE("graph executes each unary operation")
{
  const struct
  {
    const char* name;
    decltype(&popnet_abs) create;
    float input[2];
    double expected[2];
  } operations[] = {{"abs", popnet_abs, {-2, 3}, {2, 3}},
                    {"neg", popnet_neg, {-2, 3}, {2, -3}},
                    {"sqr", popnet_sqr, {-2, 3}, {4, 9}},
                    {"sqrt", popnet_sqrt, {1, 4}, {1, 2}},
                    {"exp", popnet_exp, {0, 1}, {1, 2.718281828459045}},
                    {"log", popnet_log, {1, 4}, {0, 1.3862943611198906}},
                    {"sin", popnet_sin, {0, 1}, {0, 0.8414709848078965}},
                    {"cos", popnet_cos, {0, 1}, {1, 0.5403023058681397}},
                    {"tanh", popnet_tanh, {0, 1}, {0, 0.7615941559557649}},
                    {"relu", popnet_relu, {-2, 3}, {0, 3}},
                    {"sigmoid", popnet_sigmoid, {0, 1}, {0.5f, 0.7310585786300049}}};

  for (const auto& operation : operations)
  {
    INFO("operation: " << doctest::String(operation.name));

    auto* src = popnet_empty(popnet_dtype_f32, {{1, 1, 1, 2}});

    REQUIRE(src != nullptr);
    popnet_set_data(src, operation.input, popnet_dtype_f32, 0, 2);
    auto* result = operation.create(src);

    const bool approximate = operation.create == popnet_exp || operation.create == popnet_log ||
                             operation.create == popnet_sin || operation.create == popnet_cos ||
                             operation.create == popnet_tanh || operation.create == popnet_sigmoid;
    check_graph_result(result, operation.expected, 2, approximate ? f32_math_atol : 0, approximate ? f32_math_rtol : 0);

    popnet_free(result);
    popnet_free(src);
  }
}

// @verifies REQ-0020: Forward execution dispatches each binary operation in operand order.
// @verifies REQ-0026: Forward results satisfy the operation-specific accuracy contract.
TEST_CASE("graph executes each binary operation")
{
  const struct
  {
    const char* name;
    decltype(&popnet_add) create;
    float expected[2];
  } operations[] = {{"add", popnet_add, {8, -4}},
                    {"sub", popnet_sub, {4, -12}},
                    {"mul", popnet_mul, {12, -32}},
                    {"div", popnet_div, {3, -2}}};

  auto* a = popnet_empty(popnet_dtype_f32, {{1, 1, 1, 2}});
  auto* b = popnet_empty(popnet_dtype_f32, {{1, 1, 1, 2}});

  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);

  const float a_values[] = {6, -8};
  const float b_values[] = {2, 4};
  popnet_set_data(a, a_values, popnet_dtype_f32, 0, 2);
  popnet_set_data(b, b_values, popnet_dtype_f32, 0, 2);

  for (const auto& operation : operations)
  {
    INFO("operation: " << doctest::String(operation.name));

    auto* result = operation.create(a, b);

    check_graph_result(result, operation.expected, 2);

    popnet_free(result);
  }

  popnet_free(b);
  popnet_free(a);
}

// @verifies REQ-0020: Forward execution dispatches every recorded reduction.
// @verifies REQ-0026: Forward results satisfy the operation-specific accuracy contract.
TEST_CASE("graph executes each reduction")
{
  const struct
  {
    const char* name;
    decltype(&popnet_reduce_sum) create;
    float expected;
  } reductions[] = {{"sum", popnet_reduce_sum, 9},
                    {"mean", popnet_reduce_mean, 3},
                    {"product", popnet_reduce_product, 24},
                    {"min", popnet_reduce_min, 2},
                    {"max", popnet_reduce_max, 4}};

  auto* src = popnet_empty(popnet_dtype_f32, {{1, 1, 1, 3}});

  REQUIRE(src != nullptr);

  const float values[] = {2, 3, 4};
  popnet_set_data(src, values, popnet_dtype_f32, 0, 3);

  for (const auto& reduction : reductions)
  {
    CAPTURE(reduction.name);

    auto* result = reduction.create(src, 0b1000);

    const double bound = reduction.create == popnet_reduce_sum       ? rounding_gamma(2) * 9
                         : reduction.create == popnet_reduce_mean    ? rounding_gamma(3) * 3
                         : reduction.create == popnet_reduce_product ? rounding_gamma(2) * 24
                                                                     : 0;
    check_graph_result(result, &reduction.expected, 1, bound);

    popnet_free(result);
  }

  popnet_free(src);
}

// @verifies REQ-0020: Forward execution dispatches matrix multiplication using its recorded inputs.
// @verifies REQ-0026: Forward results satisfy the operation-specific accuracy contract.
TEST_CASE("graph executes matrix multiplication")
{
  auto* a = popnet_empty(popnet_dtype_f32, {{1, 1, 2, 3}});
  auto* b = popnet_empty(popnet_dtype_f32, {{1, 1, 2, 3}});

  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);

  const float a_values[] = {1, 2, 3, 4, 5, 6};
  const float b_values[] = {1, 0, 1, 0, 2, 1};
  popnet_set_data(a, a_values, popnet_dtype_f32, 0, 6);
  popnet_set_data(b, b_values, popnet_dtype_f32, 0, 6);
  auto* result = popnet_matmul(a, b);
  const float expected[] = {4, 7, 10, 16};

  check_graph_result(result, expected, 4, 0, rounding_gamma(6));

  popnet_free(result);
  popnet_free(b);
  popnet_free(a);
}

// @verifies REQ-0020: Forward execution dispatches transpose using its recorded input.
// @verifies REQ-0026: Forward results satisfy the operation-specific accuracy contract.
TEST_CASE("graph executes transpose")
{
  auto* src = popnet_empty(popnet_dtype_f32, {{1, 1, 2, 3}});

  REQUIRE(src != nullptr);

  const float values[] = {1, 2, 3, 4, 5, 6};
  popnet_set_data(src, values, popnet_dtype_f32, 0, 6);
  auto* result = popnet_transpose(src);
  const float expected[] = {1, 4, 2, 5, 3, 6};

  check_graph_result(result, expected, 6);

  popnet_free(result);
  popnet_free(src);
}

// @verifies REQ-0018: Graph allocation starts empty and freeing a graph preserves its tensors.
TEST_CASE("graph lifetime")
{
  auto* graph = popnet_graph_alloc();
  REQUIRE(graph != nullptr);
  CHECK(graph->nodes.empty());

  auto* src = popnet_empty(popnet_dtype_f32, {{1, 1, 1, 2}});
  REQUIRE(src != nullptr);

  const float values[] = {3, -7};
  popnet_set_data(src, values, popnet_dtype_f32, 0, 2);

  REQUIRE(popnet_graph_build_forward(graph, src));
  popnet_graph_free(graph);

  float actual[2]{};
  popnet_get_data(src, actual, popnet_dtype_f32, 0, 2);

  CHECK(actual[0] == values[0]);
  CHECK(actual[1] == values[1]);

  popnet_free(src);
}

// @verifies REQ-0019: Checks dependency inclusion, deduplication, and null-output rejection.
TEST_CASE("popnet_graph build forward")
{
  popnet_graph* graph = popnet_graph_alloc();
  REQUIRE(graph != nullptr);

  popnet_shape shape{{2, 2, 1, 1}};
  popnet_tensor* a = popnet_empty(popnet_dtype_f32, shape);
  popnet_tensor* b = popnet_empty(popnet_dtype_f32, shape);

  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);

  auto check_nodes = [&](std::initializer_list<popnet_tensor*> expected) {
    REQUIRE(graph->nodes.size() == expected.size());
    for (auto* node : expected)
      CHECK(std::count(graph->nodes.begin(), graph->nodes.end(), node) == 1);
  };

  SUBCASE("nullptr")
  {
    CHECK_FALSE(popnet_graph_build_forward(graph, nullptr));
    REQUIRE(graph->nodes.empty());
  }

  SUBCASE("single node")
  {
    popnet_tensor* c = popnet_abs(a);
    REQUIRE(popnet_graph_build_forward(graph, c));

    check_nodes({a, c});

    popnet_free(c);
  }

  SUBCASE("simple binary")
  {
    popnet_tensor* c = popnet_add(a, b);
    REQUIRE(popnet_graph_build_forward(graph, c));

    check_nodes({a, b, c});

    popnet_free(c);
  }

  SUBCASE("small chain")
  {
    popnet_tensor* c = popnet_add(a, b);
    popnet_tensor* d = popnet_abs(c);
    REQUIRE(popnet_graph_build_forward(graph, d));

    check_nodes({a, b, c, d});

    popnet_free(c);
    popnet_free(d);
  }

  SUBCASE("shared subgraph")
  {
    popnet_tensor* d = popnet_abs(a);
    popnet_tensor* e = popnet_neg(a);

    REQUIRE(popnet_graph_build_forward(graph, d));
    REQUIRE(popnet_graph_build_forward(graph, e));

    check_nodes({a, d, e});

    popnet_free(d);
    popnet_free(e);
  }

  SUBCASE("diamond shape")
  {
    popnet_tensor* c1 = popnet_add(a, b);
    popnet_tensor* d1 = popnet_abs(a);
    popnet_tensor* e1 = popnet_add(c1, d1);

    REQUIRE(popnet_graph_build_forward(graph, e1));

    check_nodes({a, b, c1, d1, e1});

    popnet_free(c1);
    popnet_free(d1);
    popnet_free(e1);
  }

  SUBCASE("duplicate inputs")
  {
    popnet_tensor* f = popnet_add(a, a);
    REQUIRE(popnet_graph_build_forward(graph, f));

    check_nodes({a, f});

    popnet_free(f);
  }

  SUBCASE("long chain")
  {
    popnet_tensor* b1 = popnet_abs(a);
    popnet_tensor* c1 = popnet_neg(b1);
    popnet_tensor* d1 = popnet_sqr(c1);
    popnet_tensor* e1 = popnet_sqrt(d1);

    REQUIRE(popnet_graph_build_forward(graph, e1));

    check_nodes({a, b1, c1, d1, e1});

    popnet_free(b1);
    popnet_free(c1);
    popnet_free(d1);
    popnet_free(e1);
  }

  popnet_free(a);
  popnet_free(b);
  popnet_graph_free(graph);
}

// @verifies REQ-0020: Checks forward results for dependent operations and unchanged input-only nodes.
// @verifies REQ-0026: Forward results satisfy the operation-specific accuracy contract.
TEST_CASE("popnet_graph compute forward")
{
  popnet_graph* graph = popnet_graph_alloc();
  REQUIRE(graph != nullptr);

  popnet_shape shape{{2, 2, 1, 1}};
  popnet_tensor* a = popnet_empty(popnet_dtype_f32, shape);
  popnet_tensor* b = popnet_empty(popnet_dtype_f32, shape);

  float a_values[4] = {1, 2, 3, 4};
  float b_values[4] = {10, 20, 30, 40};

  popnet_set_data(a, a_values, popnet_dtype_f32, 0, 4);
  popnet_set_data(b, b_values, popnet_dtype_f32, 0, 4);

  SUBCASE("empty graph")
  {
    popnet_graph_compute_forward(graph);

    CHECK(graph->nodes.empty());
  }

  SUBCASE("recompute after changing inputs")
  {
    auto* sum = popnet_add(a, b);
    REQUIRE(sum != nullptr);

    auto* result = popnet_neg(sum);
    REQUIRE(result != nullptr);

    REQUIRE(popnet_graph_build_forward(graph, result));

    for (int pass = 0; pass < 2; ++pass)
    {
      const float input[4] = {float(pass + 1), -2.0f, float(pass * 10), 4.0f};
      popnet_set_data(a, input, popnet_dtype_f32, 0, 4);

      popnet_graph_compute_forward(graph);

      float actual[4]{};
      popnet_get_data(result, actual, popnet_dtype_f32, 0, 4);

      for (size_t i = 0; i < 4; ++i)
      {
        CHECK(actual[i] == -(input[i] + b_values[i]));
      }
    }

    popnet_free(result);
    popnet_free(sum);
  }

  SUBCASE("single input node with no op")
  {
    REQUIRE(popnet_graph_build_forward(graph, a));
    popnet_graph_compute_forward(graph);

    float out[4] = {};
    popnet_get_data(a, out, popnet_dtype_f32, 0, 4);

    for (int i = 0; i < 4; ++i)
    {
      CHECK(out[i] == a_values[i]);
    }
  }

  SUBCASE("multiple disconnected unary ops")
  {
    popnet_tensor* c1 = popnet_abs(a);
    popnet_tensor* c2 = popnet_neg(b);

    REQUIRE(popnet_graph_build_forward(graph, c1));
    REQUIRE(popnet_graph_build_forward(graph, c2));

    popnet_graph_compute_forward(graph);

    float c1_out[4], c2_out[4];
    popnet_get_data(c1, c1_out, popnet_dtype_f32, 0, 4);
    popnet_get_data(c2, c2_out, popnet_dtype_f32, 0, 4);

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c1_out[i] == std::abs(a_values[i]));
      CHECK(c2_out[i] == -b_values[i]);
    }

    popnet_free(c1);
    popnet_free(c2);
  }

  SUBCASE("unary then binary")
  {
    popnet_tensor* c = popnet_abs(a);
    popnet_tensor* d = popnet_add(c, b);

    REQUIRE(popnet_graph_build_forward(graph, d));
    popnet_graph_compute_forward(graph);

    float c_out[4], d_out[4];
    popnet_get_data(c, c_out, popnet_dtype_f32, 0, 4);
    popnet_get_data(d, d_out, popnet_dtype_f32, 0, 4);

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c_out[i] == std::abs(a_values[i]));
      CHECK(d_out[i] == c_out[i] + b_values[i]);
    }

    popnet_free(c);
    popnet_free(d);
  }

  SUBCASE("binary then unary")
  {
    popnet_tensor* c = popnet_add(a, b);
    popnet_tensor* d = popnet_abs(c);

    REQUIRE(popnet_graph_build_forward(graph, d));
    popnet_graph_compute_forward(graph);

    float c_out[4], d_out[4];
    popnet_get_data(c, c_out, popnet_dtype_f32, 0, 4);
    popnet_get_data(d, d_out, popnet_dtype_f32, 0, 4);

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c_out[i] == a_values[i] + b_values[i]);
      CHECK(d_out[i] == std::abs(c_out[i]));
    }

    popnet_free(c);
    popnet_free(d);
  }

  SUBCASE("diamond shape")
  {
    popnet_tensor* c = popnet_add(a, b);
    popnet_tensor* d = popnet_abs(a);
    popnet_tensor* e = popnet_add(c, d);

    REQUIRE(popnet_graph_build_forward(graph, e));
    popnet_graph_compute_forward(graph);

    float c_out[4], d_out[4], e_out[4];
    popnet_get_data(c, c_out, popnet_dtype_f32, 0, 4);
    popnet_get_data(d, d_out, popnet_dtype_f32, 0, 4);
    popnet_get_data(e, e_out, popnet_dtype_f32, 0, 4);

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c_out[i] == a_values[i] + b_values[i]);
      CHECK(d_out[i] == std::abs(a_values[i]));
      CHECK(e_out[i] == c_out[i] + d_out[i]);
    }

    popnet_free(c);
    popnet_free(d);
    popnet_free(e);
  }

  SUBCASE("shared subgraph")
  {
    popnet_tensor* c1 = popnet_abs(a);
    popnet_tensor* c2 = popnet_neg(a);
    popnet_tensor* d = popnet_add(c1, c2);

    REQUIRE(popnet_graph_build_forward(graph, d));
    popnet_graph_compute_forward(graph);

    float c1_out[4], c2_out[4], d_out[4];
    popnet_get_data(c1, c1_out, popnet_dtype_f32, 0, 4);
    popnet_get_data(c2, c2_out, popnet_dtype_f32, 0, 4);
    popnet_get_data(d, d_out, popnet_dtype_f32, 0, 4);

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c1_out[i] == std::abs(a_values[i]));
      CHECK(c2_out[i] == -a_values[i]);
      CHECK(d_out[i] == c1_out[i] + c2_out[i]);
    }

    popnet_free(c1);
    popnet_free(c2);
    popnet_free(d);
  }

  SUBCASE("stress test - long chain with shared nodes")
  {
    const int N = 512; // number of nodes in chain
    const int TENSOR_SIZE = 4;

    popnet_shape shape{{1, 1, 1, (size_t)TENSOR_SIZE}};
    popnet_tensor* base_a = popnet_empty(popnet_dtype_f32, shape);
    popnet_tensor* base_b = popnet_empty(popnet_dtype_f32, shape);

    float a_values[TENSOR_SIZE] = {1, 2, 3, 4};
    float b_values[TENSOR_SIZE] = {5, 6, 7, 8};

    popnet_set_data(base_a, a_values, popnet_dtype_f32, 0, TENSOR_SIZE);
    popnet_set_data(base_b, b_values, popnet_dtype_f32, 0, TENSOR_SIZE);

    std::vector<popnet_tensor*> nodes;
    nodes.push_back(base_a);
    nodes.push_back(base_b);

    popnet_tensor* prev = popnet_add(base_a, base_b);
    nodes.push_back(prev);

    for (int i = 1; i < N; ++i)
    {
      popnet_tensor* next;
      if (i % 2 == 0) next = popnet_abs(prev);
      else
        next = popnet_add(prev, base_a); // binary op with shared node

      nodes.push_back(next);
      prev = next;
    }

    REQUIRE(popnet_graph_build_forward(graph, prev));
    popnet_graph_compute_forward(graph);

    float last_out[TENSOR_SIZE] = {};
    popnet_get_data(prev, last_out, popnet_dtype_f32, 0, TENSOR_SIZE);

    // Positive inputs make abs an identity; every odd chain step adds base_a.
    std::vector<float> expected(TENSOR_SIZE);
    for (int j = 0; j < TENSOR_SIZE; ++j)
      expected[j] = (1 + N / 2) * a_values[j] + b_values[j];

    for (int j = 0; j < TENSOR_SIZE; ++j)
    {
      CHECK(last_out[j] == expected[j]);
    }

    for (size_t i = 2; i < nodes.size(); ++i)
    {
      popnet_free(nodes[i]);
    }

    popnet_free(base_a);
    popnet_free(base_b);
  }

  popnet_free(a);
  popnet_free(b);
  popnet_graph_free(graph);
}
