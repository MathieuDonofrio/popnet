#pragma once

#include "popnet.h"

void* popnet_alloc(popnet_type dtype, popnet_shape shape);

void popnet_free(void* ptr);
