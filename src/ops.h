#pragma once

#include "popnet.h"
#include "tensor.h"

POPNET_INLINE bool popnet_broadcastable(const popnet_tensor* a, const popnet_tensor* b)
{
  // Currently only same shape
  // TODO allow scalar broadcasting
  // TODO allow full broadcasting rules
  return a && b && (a->shape.dim[0] == b->shape.dim[0]) && (a->shape.dim[1] == b->shape.dim[1]) &&
         (a->shape.dim[2] == b->shape.dim[2]) && (a->shape.dim[3] == b->shape.dim[3]);
}

POPNET_INLINE bool popnet_valid_axis_mask(uint8_t axis_mask)
{
  static_assert(POPNET_MAX_DIMS == 4);
  return axis_mask == 0x0F; // Currently everything must be reduced
  // TODO allow any axis to be reduced
  // return axis_mask != 0               // at least one axis must be reduced
  //        && (axis_mask & ~0x0F) == 0; // mask supported dimensions supported
}

POPNET_INLINE popnet_shape popnet_reduce_shape(const popnet_tensor* a, uint8_t axis_mask)
{
  popnet_shape shape;

  for (size_t i = 0; i < POPNET_MAX_DIMS; i++)
  {
    shape.dim[i] = (axis_mask & (1 << i)) ? 1 : a->shape.dim[i];
  }

  return shape;
}

POPNET_INLINE bool popnet_matmul_compatible(const popnet_tensor* a, const popnet_tensor* b)
{
  return false; // TODO not supported yet
}
