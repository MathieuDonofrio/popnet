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

void popnet_set_data(const popnet_tensor* tensor, const void* src, popnet_dtype src_dtype, size_t offset, size_t n)
{
  POPNET_EXPECT(popnet_valid_tensor(tensor));
  POPNET_EXPECT(src != nullptr);
  POPNET_EXPECT(popnet_valid_dtype(src_dtype));
  POPNET_EXPECT(offset <= popnet_numel(tensor->shape));
  POPNET_EXPECT(n <= popnet_numel(tensor->shape) - offset);

  if (tensor->dtype == popnet_dtype_f32 && src_dtype == popnet_dtype_f32)
  {
    popnet_memcpy(static_cast<float*>(tensor->data) + offset, src, n * sizeof(float), memcpy_kind::host_to_device);
  }
  else
  {
    POPNET_UNSUPPORTED("Setting tensor data requires f32 source and tensor dtypes");
  }
}

void popnet_get_data(const popnet_tensor* tensor, void* dst, popnet_dtype dst_dtype, size_t offset, size_t n)
{
  POPNET_EXPECT(popnet_valid_tensor(tensor));
  POPNET_EXPECT(dst != nullptr);
  POPNET_EXPECT(popnet_valid_dtype(dst_dtype));
  POPNET_EXPECT(offset <= popnet_numel(tensor->shape));
  POPNET_EXPECT(n <= popnet_numel(tensor->shape) - offset);

  if (tensor->dtype == popnet_dtype_f32 && dst_dtype == popnet_dtype_f32)
  {
    popnet_memcpy(dst, static_cast<const float*>(tensor->data) + offset, n * sizeof(float),
                  memcpy_kind::device_to_host);
  }
  else
  {
    POPNET_UNSUPPORTED("Getting tensor data requires f32 tensor and destination dtypes");
  }
}
