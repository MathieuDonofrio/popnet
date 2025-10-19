#pragma once

#include "popnet.h"

#include <array>

struct popnet_tensor
{
  popnet_shape shape;

  void* data;

  std::array<popnet_tensor*, POPNET_MAX_SRC> src;

  popnet_type dtype;
  popnet_op op;
};

POPNET_INLINE bool popnet_valid_shape(popnet_shape shape)
{
  for (size_t i = 0; i < POPNET_MAX_DIMS; i++)
  {
    if (shape.dim[i] == 0) return false;
  }
  return true;
}

POPNET_INLINE bool popnet_valid_dtype(popnet_type dtype) { return dtype != POPNET_TYPE_NONE; }

POPNET_INLINE bool popnet_valid_tensor(const popnet_tensor* tensor)
{
  return tensor && tensor->dtype != POPNET_TYPE_NONE && popnet_valid_shape(tensor->shape) && tensor->data;
}
