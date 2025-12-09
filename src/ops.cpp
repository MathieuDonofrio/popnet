#include "ops.h"

popnet_tensor* popnet_add(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(b != nullptr);
  POPNET_EXPECT(popnet_shape_broadcastable(b->shape, a->shape));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_broadcast_shape(a->shape, b->shape));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_add;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_sub(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(b != nullptr);
  POPNET_EXPECT(popnet_shape_broadcastable(b->shape, a->shape));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_broadcast_shape(a->shape, b->shape));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_sub;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_mul(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(b != nullptr);
  POPNET_EXPECT(popnet_shape_broadcastable(b->shape, a->shape));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_broadcast_shape(a->shape, b->shape));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_mul;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_div(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(b != nullptr);
  POPNET_EXPECT(popnet_shape_broadcastable(b->shape, a->shape));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_broadcast_shape(a->shape, b->shape));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_div;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_abs(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_abs;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_neg(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_neg;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_sqr(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_sqr;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_sqrt(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_sqrt;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_exp(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_exp;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_log(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_log;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_sin(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_sin;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_cos(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_cos;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_tanh(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_tanh;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_relu(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_relu;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_sigmoid(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty_like(a);

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_sigmoid;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_reduce_sum(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a->shape, axis_mask));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_reduce_sum;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_reduce_mean(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a->shape, axis_mask));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_reduce_mean;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_reduce_product(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a->shape, axis_mask));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_reduce_product;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_reduce_min(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a->shape, axis_mask));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_reduce_min;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_reduce_max(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a->shape, axis_mask));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_reduce_max;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_matmul(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(a != nullptr);
  POPNET_EXPECT(b != nullptr);
  POPNET_EXPECT(popnet_matmul_shape_compatible(a->shape, b->shape));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_matmul_shape(a->shape, b->shape));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_matmul;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

popnet_tensor* popnet_transpose(popnet_tensor* a)
{
  POPNET_EXPECT(a != nullptr);

  popnet_tensor* result = popnet_empty(a->dtype, popnet_transpose_shape(a->shape));

  POPNET_EXPECT(result != nullptr);

  result->op = popnet_op_transpose;
  result->src[0] = a;

  return result;
}
