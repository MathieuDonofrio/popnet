#pragma once

#include "popnet.h"

void popnet_add_f32(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b, void* POPNET_RESTRICT c, size_t n);
void popnet_sub_f32(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b, void* POPNET_RESTRICT c, size_t n);
void popnet_mul_f32(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b, void* POPNET_RESTRICT c, size_t n);
void popnet_div_f32(const void* POPNET_RESTRICT a, const void* POPNET_RESTRICT b, void* POPNET_RESTRICT c, size_t n);

void popnet_abs_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_neg_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_sqr_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_sqrt_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_exp_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_log_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_sin_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_cos_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_tanh_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_relu_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
void popnet_sigmoid_f32(const void* POPNET_RESTRICT a, void* POPNET_RESTRICT b, size_t n);
