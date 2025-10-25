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

bool popnet_set_data(const popnet_tensor* tensor, const void* src, popnet_type src_dtype, size_t offset, size_t n)
{
  POPNET_EXPECT(popnet_valid_tensor(tensor));
  POPNET_EXPECT(src != nullptr);
  POPNET_EXPECT(popnet_valid_dtype(src_dtype));
  POPNET_EXPECT(offset + n <= popnet_nelems(tensor->shape));

  if (tensor->dtype == POPNET_TYPE_F32 && src_dtype == POPNET_TYPE_F32)
  {
    float* dst_ptr = static_cast<float*>(tensor->data) + offset;
    const float* src_ptr = static_cast<const float*>(src);
    memcpy(dst_ptr, src_ptr, n * sizeof(float)); // TODO backend copy
  }

  POPNET_UNREACHABLE(); // TODO

  return false;
}

bool popnet_get_data(const popnet_tensor* tensor, void* dst, popnet_type dst_dtype, size_t offset, size_t n)
{
  POPNET_EXPECT(popnet_valid_tensor(tensor));
  POPNET_EXPECT(dst != nullptr);
  POPNET_EXPECT(popnet_valid_dtype(dst_dtype));
  POPNET_EXPECT(offset + n <= popnet_nelems(tensor->shape));

  if (tensor->dtype == POPNET_TYPE_F32 && dst_dtype == POPNET_TYPE_F32)
  {
    const float* src_ptr = static_cast<const float*>(tensor->data) + offset;
    float* dst_ptr = static_cast<float*>(dst);
    memcpy(dst_ptr, src_ptr, n * sizeof(float)); // TODO backend copy
  }

  POPNET_UNREACHABLE(); // TODO

  return false;
}
