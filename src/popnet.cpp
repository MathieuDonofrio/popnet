#include "popnet.h"

#include <array>
#include <ranges>
#include <span>
#include <vector>

#include "allocator.h"
#include "elemwise.h"

///
/// Types
///

struct popnet_tensor
{
  popnet_shape shape;

  void* data;

  std::array<popnet_tensor*, POPNET_MAX_SRC> src;

  popnet_type dtype;
  popnet_op op;
};

struct popnet_graph
{
  std::vector<popnet_tensor*> nodes;
};

///
/// Tensor
///

POPNET_INLINE static bool popnet_valid_shape(const popnet_shape& shape)
{
  for (size_t i = 0; i < POPNET_MAX_DIMS; i++)
  {
    if (shape.dim[i] == 0) return false;
  }
  return true;
}

POPNET_INLINE static bool popnet_valid_dtype(popnet_type dtype)
{
  return dtype != POPNET_TYPE_NONE;
}

POPNET_INLINE static bool popnet_valid_tensor(const popnet_tensor* tensor)
{
  return tensor && tensor->dtype != POPNET_TYPE_NONE && popnet_valid_shape(tensor->shape) && tensor->data;
}

popnet_tensor* popnet_empty(popnet_type dtype, popnet_shape shape)
{
  POPNET_EXPECT(popnet_valid_dtype(dtype));
  POPNET_EXPECT(popnet_valid_shape(shape));

  popnet_tensor* tensor = new popnet_tensor();

  tensor->dtype = dtype;
  tensor->shape = shape;
  tensor->data = popnet_alloc(dtype, shape);

  POPNET_EXPECT(popnet_valid_tensor(tensor));

  return tensor;
}

popnet_tensor* popnet_empty_like(const popnet_tensor* tensor)
{
  if (!tensor) return nullptr;

  return popnet_empty(tensor->dtype, tensor->shape);
}

POPNET_API void popnet_free(popnet_tensor* tensor)
{
  if (!tensor) return;

  POPNET_EXPECT(popnet_valid_tensor(tensor));

  popnet_free(tensor->data);

  tensor->dtype = POPNET_TYPE_NONE;
  tensor->shape = popnet_shape{0};
  tensor->data = nullptr;

  delete tensor;
}

popnet_type popnet_dtypeof(const popnet_tensor* tensor)
{
  return tensor ? tensor->dtype : popnet_type::POPNET_TYPE_NONE;
}

popnet_shape popnet_shapeof(const popnet_tensor* tensor)
{
  return tensor ? tensor->shape : popnet_shape{0};
}

size_t popnet_nelems(popnet_shape shape)
{
  size_t nelems = 1;

  for (size_t i = 0; i < POPNET_MAX_DIMS; i++)
  {
    nelems *= shape.dim[i];
  }

  POPNET_EXPECT(nelems > 0);

  return nelems;
}

///
/// Graph
///

popnet_graph* popnet_graph_alloc()
{
  return new popnet_graph();
}

void popnet_graph_free(popnet_graph* graph)
{
  delete graph;
}

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
  // TODO
}

///
/// Operations
///

POPNET_INLINE static bool popnet_broadcastable(const popnet_tensor* a, const popnet_tensor* b)
{
  // TODO allow scalar broadcasting
  // TODO allow full broadcasting rules
  return a && b && (a->shape.dim[0] == b->shape.dim[0]) && (a->shape.dim[1] == b->shape.dim[1]) &&
         (a->shape.dim[2] == b->shape.dim[2]) && (a->shape.dim[3] == b->shape.dim[3]);
}

popnet_tensor* popnet_add(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(popnet_broadcastable(b, a));

  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_ADD;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_sub(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(popnet_broadcastable(b, a));

  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_SUB;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_mul(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(popnet_broadcastable(b, a));

  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_MUL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_div(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(popnet_broadcastable(b, a));

  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_DIV;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_abs(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_ABS;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_neg(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_NEG;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_sqr(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_SQR;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_sqrt(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_SQRT;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_exp(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_EXP;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_log(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_LOG;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_sin(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_SIN;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_cos(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_COS;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_tanh(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_TANH;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_relu(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_RELU;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_sigmoid(popnet_tensor* a)
{
  popnet_tensor* result = popnet_empty_like(a);

  result->op = POPNET_OP_SIGMOID;
  result->src[0] = a;

  return result;
}

POPNET_INLINE static bool popnet_valid_axis_mask(uint8_t axis_mask)
{
  return axis_mask != 0               // at least one axis must be reduced
         && (axis_mask & ~0x0F) == 0; // only 4 dimensions supported
}

POPNET_INLINE static popnet_shape popnet_reduce_shape(const popnet_tensor* a, uint8_t axis_mask)
{
  popnet_shape shape;

  for (size_t i = 0; i < POPNET_MAX_DIMS; i++)
  {
    shape.dim[i] = (axis_mask & (1 << i)) ? 1 : a->shape.dim[i];
  }

  return shape;
}

popnet_tensor* popnet_sum_row(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a, axis_mask));

  result->op = POPNET_OP_SUM;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_mean_row(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a, axis_mask));

  result->op = POPNET_OP_MEAN;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_max_row(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a, axis_mask));

  result->op = POPNET_OP_MAX;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_min_row(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a, axis_mask));

  result->op = POPNET_OP_MIN;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_matmul(popnet_tensor* a, popnet_tensor* b)
{
  // TODO assert shapes compatible

  popnet_tensor* result = nullptr; // TODO create with correct shape

  result->op = POPNET_OP_MATMUL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_transpose(popnet_tensor* a)
{
  popnet_tensor* result = nullptr; // TODO create with correct shape

  result->op = POPNET_OP_TRANSPOSE;
  result->src[0] = a;

  return result;
}
