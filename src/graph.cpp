#include "graph.h"

#include "elemwise.h"
#include "reduce.h"
#include "tensor.h"

popnet_graph* popnet_graph_alloc() { return new popnet_graph(); }

void popnet_graph_free(popnet_graph* graph) { delete graph; }

void popnet_graph_build_forward(popnet_graph* graph, popnet_tensor* tensor)
{
  if (!tensor) return;

  for (auto node : graph->nodes)
  {
    if (node == tensor) return;
  }

  for (auto src : tensor->src)
  {
    if (src)
    {
      popnet_graph_build_forward(graph, src);
    }
  }

  graph->nodes.push_back(tensor);
}

void popnet_graph_compute_forward(popnet_graph* graph)
{
  if (!graph) return;

  for (auto* node : graph->nodes)
  {
    switch (node->op)
    {
    case POPNET_OP_NONE: break; // Input or constant node
    case POPNET_OP_ADD: popnet_compute_add(node); break;
    case POPNET_OP_SUB: popnet_compute_sub(node); break;
    case POPNET_OP_MUL: popnet_compute_mul(node); break;
    case POPNET_OP_DIV: popnet_compute_div(node); break;
    case POPNET_OP_ABS: popnet_compute_abs(node); break;
    case POPNET_OP_NEG: popnet_compute_neg(node); break;
    case POPNET_OP_SQR: popnet_compute_sqr(node); break;
    case POPNET_OP_SQRT: popnet_compute_sqrt(node); break;
    case POPNET_OP_EXP: popnet_compute_exp(node); break;
    case POPNET_OP_LOG: popnet_compute_log(node); break;
    case POPNET_OP_SIN: popnet_compute_sin(node); break;
    case POPNET_OP_COS: popnet_compute_cos(node); break;
    case POPNET_OP_TANH: popnet_compute_tanh(node); break;
    case POPNET_OP_RELU: popnet_compute_relu(node); break;
    case POPNET_OP_SIGMOID: popnet_compute_sigmoid(node); break;
    case POPNET_OP_REDUCE_SUM: popnet_compute_reduce_sum(node); break;
    case POPNET_OP_REDUCE_MEAN: popnet_compute_reduce_mean(node); break;
    case POPNET_OP_REDUCE_PRODUCT: popnet_compute_reduce_product(node); break;
    case POPNET_OP_REDUCE_MIN: popnet_compute_reduce_min(node); break;
    case POPNET_OP_REDUCE_MAX: popnet_compute_reduce_max(node); break;
    case POPNET_OP_MATMUL:
      POPNET_UNREACHABLE(); // TODO
      break;
    case POPNET_OP_TRANSPOSE:
      POPNET_UNREACHABLE(); // TODO
      break;
    default: POPNET_UNREACHABLE(); break;
    }
  }
}
