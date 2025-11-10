#pragma once

#include "popnet.h"

enum class MemcpyKind
{
  HostToHost,
  HostToDevice,
  DeviceToHost,
  DeviceToDevice
};

void* popnet_alloc(popnet_type dtype, popnet_shape shape) noexcept;

void popnet_free(void* ptr) noexcept;

void popnet_memcpy(void* POPNET_RESTRICT dst, const void* POPNET_RESTRICT src, size_t count, MemcpyKind kind) noexcept;
