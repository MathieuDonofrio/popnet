#pragma once

#include <array>

#include "popnet.h"

struct popnet_tensor
{
  popnet_shape shape;

  void* data;

  // Borrowed dependencies, fixed at creation; unused entries are null.
  std::array<popnet_tensor*, POPNET_MAX_SRC> src;

  popnet_dtype dtype;
  popnet_op op;
};

POPNET_INLINE bool operator==(const popnet_shape& a, const popnet_shape& b)
{
  static_assert(POPNET_MAX_DIMS == 4);
  return a.dim[0] == b.dim[0] && a.dim[1] == b.dim[1] && a.dim[2] == b.dim[2] && a.dim[3] == b.dim[3];
}

POPNET_INLINE size_t popnet_inner_numel(popnet_shape shape, size_t split_axis)
{
  size_t numel = 1;
  for (size_t i = split_axis; i < POPNET_MAX_DIMS; ++i)
  {
    numel *= shape.dim[i];
  }
  return numel;
}

POPNET_INLINE bool popnet_valid_shape(popnet_shape shape)
{
  static_assert(POPNET_MAX_DIMS == 4);
  return shape.dim[0] != 0 && shape.dim[1] != 0 && shape.dim[2] != 0 && shape.dim[3] != 0;
}

POPNET_INLINE bool popnet_valid_dtype(popnet_dtype dtype) { return dtype == popnet_dtype_f32; }

POPNET_INLINE bool popnet_valid_tensor(const popnet_tensor* tensor)
{
  return tensor && popnet_valid_dtype(tensor->dtype) && popnet_valid_shape(tensor->shape) && tensor->data;
}
