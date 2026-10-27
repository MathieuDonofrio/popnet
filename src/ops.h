#pragma once

#include "popnet.h"
#include "tensor.h"

POPNET_INLINE bool popnet_shape_broadcastable(const popnet_shape a, const popnet_shape b)
{
  for (size_t i = 0; i < POPNET_MAX_DIMS; ++i)
  {
    if (a.dim[i] != b.dim[i] && a.dim[i] != 1 && b.dim[i] != 1)
    {
      return false;
    }
  }

  return true;
}

POPNET_INLINE bool popnet_valid_axis_mask(const uint8_t axis_mask)
{
  static_assert(POPNET_MAX_DIMS == 4);
  return axis_mask != 0               // at least one axis must be reduced
         && (axis_mask & ~0x0F) == 0; // mask uses supported dimensions
}

POPNET_INLINE bool popnet_matmul_shape_compatible(const popnet_shape a, const popnet_shape b)
{
  static_assert(POPNET_MAX_DIMS == 4);
  // a shape = (a, b, m, k)
  // b shape = (1, 1, n, k)
  return (b.dim[0] == 1 && b.dim[1] == 1 && a.dim[3] == b.dim[3]);
}

POPNET_INLINE popnet_shape popnet_broadcast_shape(const popnet_shape a, const popnet_shape b)
{
  popnet_shape shape{};

  for (size_t i = 0; i < POPNET_MAX_DIMS; ++i)
  {
    const size_t ad = a.dim[i];
    const size_t bd = b.dim[i];

    if (ad != bd && ad != 1 && bd != 1) [[unlikely]]
    {
      POPNET_UNSUPPORTED("Broadcasting requires matching dimensions or a dimension of size 1");
    }

    shape.dim[i] = (ad > bd ? ad : bd);
  }

  return shape;
}

POPNET_INLINE popnet_shape popnet_reduce_shape(const popnet_shape a, const uint8_t axis_mask)
{
  popnet_shape shape;

  for (size_t i = 0; i < POPNET_MAX_DIMS; ++i)
  {
    shape.dim[i] = (axis_mask & (1 << i)) ? 1 : a.dim[i];
  }

  return shape;
}

POPNET_INLINE popnet_shape popnet_transpose_shape(const popnet_shape a)
{
  static_assert(POPNET_MAX_DIMS == 4);
  return {a.dim[0], a.dim[1], a.dim[3], a.dim[2]};
}

POPNET_INLINE popnet_shape popnet_matmul_shape(const popnet_shape a, const popnet_shape b)
{
  static_assert(POPNET_MAX_DIMS == 4);
  // a shape = (a, b, m, k)
  // b shape = (1, 1, n, k)
  // c shape = (a, b, m, n)
  return {a.dim[0], a.dim[1], a.dim[2], b.dim[2]};
}
