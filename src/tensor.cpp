#include "tensor.h"

#include "allocator.h"

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

void popnet_free(popnet_tensor* tensor)
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

popnet_shape popnet_shapeof(const popnet_tensor* tensor) { return tensor ? tensor->shape : popnet_shape{0}; }
