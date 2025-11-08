#include "ops.h"

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

popnet_tensor* popnet_reduce_sum(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a, axis_mask));

  result->op = POPNET_OP_SUM;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_reduce_mean(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a, axis_mask));

  result->op = POPNET_OP_MEAN;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_reduce_product(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a, axis_mask));

  result->op = POPNET_OP_PRODUCT;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_reduce_min(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a, axis_mask));

  result->op = POPNET_OP_MIN;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_reduce_max(popnet_tensor* a, uint8_t axis_mask)
{
  POPNET_EXPECT(popnet_valid_axis_mask(axis_mask));

  popnet_tensor* result = popnet_empty(a->dtype, popnet_reduce_shape(a, axis_mask));

  result->op = POPNET_OP_MAX;
  result->src[0] = a;

  return result;
}

popnet_tensor* popnet_matmul(popnet_tensor* a, popnet_tensor* b)
{
  POPNET_EXPECT(popnet_matmul_compatible(a, b));

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
