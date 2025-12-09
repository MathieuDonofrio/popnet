#pragma once

#include <array>

#include "popnet.h"

struct popnet_tensor
{
  popnet_shape shape;

  void* data;

  std::array<popnet_tensor*, POPNET_MAX_SRC> src;

  popnet_type dtype;
  popnet_op op;
};

POPNET_INLINE bool operator==(const popnet_shape& a, const popnet_shape& b)
{
  static_assert(POPNET_MAX_DIMS == 4);
  return a.dim[0] == b.dim[0] && a.dim[1] == b.dim[1] && a.dim[2] == b.dim[2] && a.dim[3] == b.dim[3];
}

POPNET_INLINE bool popnet_valid_shape(popnet_shape shape)
{
  static_assert(POPNET_MAX_DIMS == 4);
  return shape.dim[0] != 0 && shape.dim[1] != 0 && shape.dim[2] != 0 && shape.dim[3] != 0;
}

POPNET_INLINE bool popnet_valid_dtype(popnet_type dtype) { return dtype != popnet_type_none; }

POPNET_INLINE bool popnet_valid_tensor(const popnet_tensor* tensor)
{
  return tensor && tensor->dtype != popnet_type_none && popnet_valid_shape(tensor->shape) && tensor->data;
}
