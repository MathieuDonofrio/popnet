#include "tensor.h"

#include <new>

#include "memory.h"

popnet_tensor* popnet_empty(popnet_dtype dtype, popnet_shape shape)
{
  POPNET_EXPECT(popnet_valid_dtype(dtype));
  POPNET_EXPECT(popnet_valid_shape(shape));

  popnet_tensor* tensor = new (std::nothrow) popnet_tensor();

  if (!tensor) return nullptr;

  void* data = popnet_alloc(dtype, shape);

  if (!data)
  {
    delete tensor;
    return nullptr;
  }

  tensor->shape = shape;
  tensor->data = data;
  tensor->dtype = dtype;
  tensor->op = popnet_op_none;

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

  popnet_free(tensor->data);

  tensor->shape = popnet_shape{0};
  tensor->data = nullptr;
  tensor->dtype = popnet_dtype_none;

  delete tensor;
}

popnet_dtype popnet_dtypeof(const popnet_tensor* tensor) { return tensor ? tensor->dtype : popnet_dtype_none; }

popnet_shape popnet_shapeof(const popnet_tensor* tensor) { return tensor ? tensor->shape : popnet_shape{0}; }

popnet_op popnet_opof(const popnet_tensor* tensor) { return tensor ? tensor->op : popnet_op_none; }

namespace
{
constexpr size_t transfer_element_size(popnet_dtype dtype)
{
  switch (dtype)
  {
  case popnet_dtype_f32: return sizeof(float);
  case popnet_dtype_i8: return sizeof(int8_t);
  default: POPNET_UNSUPPORTED("Unsupported tensor transfer dtype"); return 0;
  }
}
} // namespace

void popnet_set_data(const popnet_tensor* tensor, const void* src, size_t offset, size_t n)
{
  POPNET_EXPECT(popnet_valid_tensor(tensor));
  POPNET_EXPECT(src != nullptr);
  POPNET_EXPECT(offset <= popnet_numel(tensor->shape));
  POPNET_EXPECT(n <= popnet_numel(tensor->shape) - offset);

  const size_t element_size = transfer_element_size(tensor->dtype);
  auto* dst = static_cast<unsigned char*>(tensor->data) + offset * element_size;
  popnet_memcpy(dst, src, n * element_size, memcpy_kind::host_to_device);
}

void popnet_get_data(const popnet_tensor* tensor, void* dst, size_t offset, size_t n)
{
  POPNET_EXPECT(popnet_valid_tensor(tensor));
  POPNET_EXPECT(dst != nullptr);
  POPNET_EXPECT(offset <= popnet_numel(tensor->shape));
  POPNET_EXPECT(n <= popnet_numel(tensor->shape) - offset);

  const size_t element_size = transfer_element_size(tensor->dtype);
  const auto* src = static_cast<const unsigned char*>(tensor->data) + offset * element_size;
  popnet_memcpy(dst, src, n * element_size, memcpy_kind::device_to_host);
}
