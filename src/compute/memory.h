#pragma once

#include "popnet.h"

enum class memcpy_kind
{
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device
};

void* popnet_alloc(popnet_type dtype, popnet_shape shape) noexcept;

void popnet_free(void* ptr) noexcept;

void popnet_memcpy(void* POPNET_RESTRICT dst, const void* POPNET_RESTRICT src, size_t count, memcpy_kind kind) noexcept;
