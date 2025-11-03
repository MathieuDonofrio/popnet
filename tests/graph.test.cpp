#include <doctest/doctest.h>

#include "graph.h"
#include "ops.h"
#include "tensor.h"

TEST_CASE("popnet_graph build forward")
{
  popnet_graph* graph = popnet_graph_alloc();
  REQUIRE(graph != nullptr);

  popnet_shape shape{{2, 2, 1, 1}};
  popnet_tensor* a = popnet_empty(POPNET_TYPE_F32, shape);
  popnet_tensor* b = popnet_empty(POPNET_TYPE_F32, shape);
  popnet_tensor* c = popnet_empty(POPNET_TYPE_F32, shape);

  SUBCASE("nullptr")
  {
    popnet_graph_build_forward(graph, nullptr);
    REQUIRE(graph->nodes.empty());
  }

  SUBCASE("single node")
  {
    popnet_tensor* c = popnet_abs(a);
    popnet_graph_build_forward(graph, c);

    REQUIRE(graph->nodes.size() == 2);
    CHECK(graph->nodes[0] == a);
    CHECK(graph->nodes[1] == c);

    popnet_free(c);
  }

  SUBCASE("simple binary")
  {
    popnet_tensor* c = popnet_add(a, b);
    popnet_graph_build_forward(graph, c);

    REQUIRE(graph->nodes.size() == 3);
    CHECK(graph->nodes[0] == a);
    CHECK(graph->nodes[1] == b);
    CHECK(graph->nodes[2] == c);

    popnet_free(c);
  }

  SUBCASE("small chain")
  {
    popnet_tensor* c = popnet_add(a, b);
    popnet_tensor* d = popnet_abs(c);
    popnet_graph_build_forward(graph, d);

    REQUIRE(graph->nodes.size() == 4);
    CHECK(graph->nodes[0] == a);
    CHECK(graph->nodes[1] == b);
    CHECK(graph->nodes[2] == c);
    CHECK(graph->nodes[3] == d);

    popnet_free(c);
    popnet_free(d);
  }

  SUBCASE("shared subgraph")
  {
    popnet_tensor* d = popnet_abs(a);
    popnet_tensor* e = popnet_neg(a);

    popnet_graph_build_forward(graph, d);
    popnet_graph_build_forward(graph, e);

    REQUIRE(graph->nodes.size() == 3);
    CHECK(graph->nodes[0] == a);
    CHECK((graph->nodes[1] == d || graph->nodes[1] == e));
    CHECK((graph->nodes[2] == d || graph->nodes[2] == e));
    REQUIRE(graph->nodes[1] != graph->nodes[2]);

    popnet_free(d);
    popnet_free(e);
  }

  SUBCASE("diamond shape")
  {
    popnet_tensor* c1 = popnet_add(a, b);
    popnet_tensor* d1 = popnet_abs(a);
    popnet_tensor* e1 = popnet_add(c1, d1);

    popnet_graph_build_forward(graph, e1);

    REQUIRE(graph->nodes.size() == 5);
    CHECK(graph->nodes[0] == a);
    CHECK(graph->nodes[1] == b);
    CHECK((graph->nodes[2] == c1 || graph->nodes[2] == d1));
    CHECK((graph->nodes[3] == c1 || graph->nodes[3] == d1));
    CHECK(graph->nodes[4] == e1);

    popnet_free(c1);
    popnet_free(d1);
    popnet_free(e1);
  }

  SUBCASE("duplicate inputs")
  {
    popnet_tensor* f = popnet_add(a, a);
    popnet_graph_build_forward(graph, f);

    REQUIRE(graph->nodes.size() == 2);
    CHECK(graph->nodes[0] == a);
    CHECK(graph->nodes[1] == f);

    popnet_free(f);
  }

  SUBCASE("long chain")
  {
    popnet_tensor* b1 = popnet_abs(a);
    popnet_tensor* c1 = popnet_neg(b1);
    popnet_tensor* d1 = popnet_sqr(c1);
    popnet_tensor* e1 = popnet_sqrt(d1);

    popnet_graph_build_forward(graph, e1);

    REQUIRE(graph->nodes.size() == 5);
    CHECK(graph->nodes[0] == a);
    CHECK(graph->nodes[1] == b1);
    CHECK(graph->nodes[2] == c1);
    CHECK(graph->nodes[3] == d1);
    CHECK(graph->nodes[4] == e1);

    popnet_free(b1);
    popnet_free(c1);
    popnet_free(d1);
    popnet_free(e1);
  }

  popnet_free(a);
  popnet_free(b);
  popnet_graph_free(graph);
}

TEST_CASE("popnet_graph compute forward")
{
  popnet_graph* graph = popnet_graph_alloc();
  REQUIRE(graph != nullptr);

  popnet_shape shape{{2, 2, 1, 1}};
  popnet_tensor* a = popnet_empty(POPNET_TYPE_F32, shape);
  popnet_tensor* b = popnet_empty(POPNET_TYPE_F32, shape);

  float a_values[4] = {1, 2, 3, 4};
  float b_values[4] = {10, 20, 30, 40};

  REQUIRE(popnet_set_data(a, a_values, POPNET_TYPE_F32, 0, 4));
  REQUIRE(popnet_set_data(b, b_values, POPNET_TYPE_F32, 0, 4));

  SUBCASE("null graph pointer") { popnet_graph_compute_forward(nullptr); }

  SUBCASE("empty graph") { popnet_graph_compute_forward(graph); }

  SUBCASE("single input node with no op")
  {
    popnet_graph_build_forward(graph, a);
    popnet_graph_compute_forward(graph);

    float out[4] = {};
    REQUIRE(popnet_get_data(a, out, POPNET_TYPE_F32, 0, 4));

    for (int i = 0; i < 4; ++i)
    {
      CHECK(out[i] == doctest::Approx(a_values[i]).epsilon(1e-4));
    }
  }

  SUBCASE("multiple disconnected unary ops")
  {
    popnet_tensor* c1 = popnet_abs(a);
    popnet_tensor* c2 = popnet_neg(b);

    popnet_graph_build_forward(graph, c1);
    popnet_graph_build_forward(graph, c2);

    popnet_graph_compute_forward(graph);

    float c1_out[4], c2_out[4];
    REQUIRE(popnet_get_data(c1, c1_out, POPNET_TYPE_F32, 0, 4));
    REQUIRE(popnet_get_data(c2, c2_out, POPNET_TYPE_F32, 0, 4));

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c1_out[i] == doctest::Approx(std::abs(a_values[i])).epsilon(1e-4));
      CHECK(c2_out[i] == doctest::Approx(-b_values[i]).epsilon(1e-4));
    }

    popnet_free(c1);
    popnet_free(c2);
  }

  SUBCASE("unary then binary")
  {
    popnet_tensor* c = popnet_abs(a);
    popnet_tensor* d = popnet_add(c, b);

    popnet_graph_build_forward(graph, d);
    popnet_graph_compute_forward(graph);

    float c_out[4], d_out[4];
    REQUIRE(popnet_get_data(c, c_out, POPNET_TYPE_F32, 0, 4));
    REQUIRE(popnet_get_data(d, d_out, POPNET_TYPE_F32, 0, 4));

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c_out[i] == doctest::Approx(std::abs(a_values[i])).epsilon(1e-4));
      CHECK(d_out[i] == doctest::Approx(c_out[i] + b_values[i]).epsilon(1e-4));
    }

    popnet_free(c);
    popnet_free(d);
  }

  SUBCASE("binary then unary")
  {
    popnet_tensor* c = popnet_add(a, b);
    popnet_tensor* d = popnet_abs(c);

    popnet_graph_build_forward(graph, d);
    popnet_graph_compute_forward(graph);

    float c_out[4], d_out[4];
    REQUIRE(popnet_get_data(c, c_out, POPNET_TYPE_F32, 0, 4));
    REQUIRE(popnet_get_data(d, d_out, POPNET_TYPE_F32, 0, 4));

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c_out[i] == doctest::Approx(a_values[i] + b_values[i]).epsilon(1e-4));
      CHECK(d_out[i] == doctest::Approx(std::abs(c_out[i])).epsilon(1e-4));
    }

    popnet_free(c);
    popnet_free(d);
  }

  SUBCASE("diamond shape")
  {
    popnet_tensor* c = popnet_add(a, b);
    popnet_tensor* d = popnet_abs(a);
    popnet_tensor* e = popnet_add(c, d);

    popnet_graph_build_forward(graph, e);
    popnet_graph_compute_forward(graph);

    float c_out[4], d_out[4], e_out[4];
    REQUIRE(popnet_get_data(c, c_out, POPNET_TYPE_F32, 0, 4));
    REQUIRE(popnet_get_data(d, d_out, POPNET_TYPE_F32, 0, 4));
    REQUIRE(popnet_get_data(e, e_out, POPNET_TYPE_F32, 0, 4));

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c_out[i] == doctest::Approx(a_values[i] + b_values[i]).epsilon(1e-4));
      CHECK(d_out[i] == doctest::Approx(std::abs(a_values[i])).epsilon(1e-4));
      CHECK(e_out[i] == doctest::Approx(c_out[i] + d_out[i]).epsilon(1e-4));
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

    popnet_graph_build_forward(graph, d);
    popnet_graph_compute_forward(graph);

    float c1_out[4], c2_out[4], d_out[4];
    REQUIRE(popnet_get_data(c1, c1_out, POPNET_TYPE_F32, 0, 4));
    REQUIRE(popnet_get_data(c2, c2_out, POPNET_TYPE_F32, 0, 4));
    REQUIRE(popnet_get_data(d, d_out, POPNET_TYPE_F32, 0, 4));

    for (int i = 0; i < 4; ++i)
    {
      CHECK(c1_out[i] == doctest::Approx(std::abs(a_values[i])).epsilon(1e-4));
      CHECK(c2_out[i] == doctest::Approx(-a_values[i]).epsilon(1e-4));
      CHECK(d_out[i] == doctest::Approx(c1_out[i] + c2_out[i]).epsilon(1e-4));
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
    popnet_tensor* base_a = popnet_empty(POPNET_TYPE_F32, shape);
    popnet_tensor* base_b = popnet_empty(POPNET_TYPE_F32, shape);

    float a_values[TENSOR_SIZE] = {1, 2, 3, 4};
    float b_values[TENSOR_SIZE] = {5, 6, 7, 8};

    REQUIRE(popnet_set_data(base_a, a_values, POPNET_TYPE_F32, 0, TENSOR_SIZE));
    REQUIRE(popnet_set_data(base_b, b_values, POPNET_TYPE_F32, 0, TENSOR_SIZE));

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

    popnet_graph_build_forward(graph, prev);
    popnet_graph_compute_forward(graph);

    float last_out[TENSOR_SIZE] = {};
    REQUIRE(popnet_get_data(prev, last_out, POPNET_TYPE_F32, 0, TENSOR_SIZE));

    std::vector<float> expected(TENSOR_SIZE);
    for (int j = 0; j < TENSOR_SIZE; ++j)
    {
      expected[j] = a_values[j] + b_values[j]; // first add
    }

    for (int i = 1; i < N; ++i)
    {
      if (i % 2 == 0)
      {
        for (int j = 0; j < TENSOR_SIZE; ++j)
        {
          expected[j] = std::abs(expected[j]);
        }
      }
      else
      {
        for (int j = 0; j < TENSOR_SIZE; ++j)
        {
          expected[j] += a_values[j];
        }
      }
    }

    for (int j = 0; j < TENSOR_SIZE; ++j)
    {
      CHECK(last_out[j] == doctest::Approx(expected[j]).epsilon(1e-4));
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
