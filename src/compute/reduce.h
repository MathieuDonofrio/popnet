#pragma once

#include "popnet.h"

void popnet_compute_reduce_sum(popnet_tensor* dst);
void popnet_compute_reduce_mean(popnet_tensor* dst);
void popnet_compute_reduce_product(popnet_tensor* dst);
void popnet_compute_reduce_min(popnet_tensor* dst);
void popnet_compute_reduce_max(popnet_tensor* dst);