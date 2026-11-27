#pragma once

#include "popnet.h"

constexpr uint64_t dtype_pair_key(popnet_dtype src, popnet_dtype dst)
{
  static_assert(sizeof(popnet_dtype) <= sizeof(uint32_t));
  return (static_cast<uint64_t>(static_cast<uint32_t>(src)) << 32) | static_cast<uint32_t>(dst);
}

template <popnet_dtype Src, popnet_dtype Dst> constexpr uint64_t dtype_pair_key() { return dtype_pair_key(Src, Dst); }
