#include "graph.h"

#include <new>
#include <stdexcept>

#include "elemwise.h"
#include "linalg.h"
#include "reduce.h"
#include "tensor.h"

popnet_graph* popnet_graph_alloc() { return new (std::nothrow) popnet_graph(); }

void popnet_graph_free(popnet_graph* graph) { delete graph; }

bool popnet_graph_build_forward(popnet_graph* graph, popnet_tensor* tensor)
{
  POPNET_EXPECT(graph != nullptr);
  if (!tensor) return false;

  for (auto node : graph->nodes)
  {
    if (node == tensor) return true;
  }

  for (auto src : tensor->src)
  {
    if (src)
    {
      if (!popnet_graph_build_forward(graph, src)) return false;
    }
  }

  try
  {
    // Dependencies must precede their consumer because computation walks nodes in insertion order.
    graph->nodes.push_back(tensor);
  }
  catch (const std::bad_alloc&)
  {
    return false;
  }
  catch (const std::length_error&)
  {
    return false;
  }

  return true;
}

void popnet_graph_compute_forward(popnet_graph* graph)
{
  if (!graph) return;

  for (auto* node : graph->nodes)
  {
    switch (node->op)
    {
    case popnet_op_none: break; // Input or constant node
    case popnet_op_add: popnet_compute_add(node); break;
    case popnet_op_sub: popnet_compute_sub(node); break;
    case popnet_op_mul: popnet_compute_mul(node); break;
    case popnet_op_div: popnet_compute_div(node); break;
    case popnet_op_abs: popnet_compute_abs(node); break;
    case popnet_op_neg: popnet_compute_neg(node); break;
    case popnet_op_sqr: popnet_compute_sqr(node); break;
    case popnet_op_sqrt: popnet_compute_sqrt(node); break;
    case popnet_op_exp: popnet_compute_exp(node); break;
    case popnet_op_log: popnet_compute_log(node); break;
    case popnet_op_sin: popnet_compute_sin(node); break;
    case popnet_op_cos: popnet_compute_cos(node); break;
    case popnet_op_tanh: popnet_compute_tanh(node); break;
    case popnet_op_relu: popnet_compute_relu(node); break;
    case popnet_op_sigmoid: popnet_compute_sigmoid(node); break;
    case popnet_op_reduce_sum: popnet_compute_reduce_sum(node); break;
    case popnet_op_reduce_mean: popnet_compute_reduce_mean(node); break;
    case popnet_op_reduce_product: popnet_compute_reduce_product(node); break;
    case popnet_op_reduce_min: popnet_compute_reduce_min(node); break;
    case popnet_op_reduce_max: popnet_compute_reduce_max(node); break;
    case popnet_op_matmul: popnet_compute_matmul(node); break;
    case popnet_op_transpose: popnet_compute_transpose(node); break;
    default: POPNET_UNSUPPORTED("Unsupported graph operation"); break;
    }
  }
}
