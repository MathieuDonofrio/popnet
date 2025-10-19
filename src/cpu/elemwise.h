#pragma once

#include "popnet.h"

void popnet_compute_add(popnet_tensor* dst);
void popnet_compute_sub(popnet_tensor* dst);
void popnet_compute_mul(popnet_tensor* dst);
void popnet_compute_div(popnet_tensor* dst);
void popnet_compute_abs(popnet_tensor* dst);
void popnet_compute_neg(popnet_tensor* dst);
void popnet_compute_sqr(popnet_tensor* dst);
void popnet_compute_sqrt(popnet_tensor* dst);
void popnet_compute_exp(popnet_tensor* dst);
void popnet_compute_log(popnet_tensor* dst);
void popnet_compute_sin(popnet_tensor* dst);
void popnet_compute_cos(popnet_tensor* dst);
void popnet_compute_tanh(popnet_tensor* dst);
void popnet_compute_relu(popnet_tensor* dst);
void popnet_compute_sigmoid(popnet_tensor* dst);
