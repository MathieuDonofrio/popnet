#ifndef POPNET_H
#define POPNET_H

#include <stddef.h>
#include <stdint.h>

#ifdef POPNET_SHARED
#ifdef _MSC_VER
#ifdef POPNET_BUILD
#define POPNET_API __declspec(dllexport)
#else
#define POPNET_API __declspec(dllimport)
#endif
#else
#define POPNET_API __attribute__((visibility("default")))
#endif
#else
#define POPNET_API
#endif

#ifdef __cplusplus
#ifdef _MSC_VER
#define POPNET_RESTRICT __restrict
#elif defined(__clang__) || defined(__GNUC__)
#define POPNET_RESTRICT __restrict__
#else
#define POPNET_RESTRICT
#endif
#else
#define POPNET_RESTRICT restrict
#endif

#ifdef _MSC_VER
#define POPNET_INLINE __forceinline
#define POPNET_NO_INLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define POPNET_INLINE inline __attribute__((always_inline))
#define POPNET_NO_INLINE __attribute__((noinline))
#else
#define POPNET_INLINE inline
#define POPNET_NO_INLINE
#endif

#ifndef NDEBUG
#include <assert.h>
#define POPNET_EXPECT(cond) assert(cond)
#define POPNET_UNREACHABLE() assert(false)
#else
#ifdef _MSC_VER
#define POPNET_EXPECT(cond) __assume(cond)
#define POPNET_UNREACHABLE() __assume(0)
#elif defined(__GNUC__) || defined(__clang__)
#define POPNET_EXPECT(cond) __builtin_assume(cond)
#define POPNET_UNREACHABLE() __builtin_unreachable()
#else
#define POPNET_EXPECT(cond) ((void)0)
#define POPNET_UNREACHABLE() ((void)0)
#endif
#endif

#define POPNET_MAX_DIMS 4
#define POPNET_MAX_SRC 8

#ifdef __cplusplus
extern "C"
{
#endif

///
/// Tensor
///

typedef enum
{
  popnet_type_none = 0,
  popnet_type_f32,
} popnet_type;

typedef enum
{
  popnet_op_none = 0,

  popnet_op_add,
  popnet_op_sub,
  popnet_op_mul,
  popnet_op_div,

  popnet_op_abs,
  popnet_op_neg,
  popnet_op_sqr,
  popnet_op_sqrt,
  popnet_op_exp,
  popnet_op_log,
  popnet_op_sin,
  popnet_op_cos,
  popnet_op_tanh,
  popnet_op_relu,
  popnet_op_sigmoid,

  popnet_op_reduce_sum,
  popnet_op_reduce_mean,
  popnet_op_reduce_product,
  popnet_op_reduce_min,
  popnet_op_reduce_max,

  popnet_op_matmul,
  popnet_op_transpose,
} popnet_op;

typedef struct popnet_shape
{
  size_t dim[POPNET_MAX_DIMS];
} popnet_shape;

const popnet_shape popnet_scalar_shape = {.dim = {1, 1, 1, 1}};

POPNET_INLINE size_t popnet_numel(popnet_shape shape)
{
  static_assert(POPNET_MAX_DIMS == 4);
  const size_t size = shape.dim[0] * shape.dim[1] * shape.dim[2] * shape.dim[3];
  POPNET_EXPECT(size > 0);
  return size;
}

typedef struct popnet_tensor popnet_tensor;

POPNET_API popnet_tensor* popnet_empty(popnet_type dtype, popnet_shape shape);
POPNET_API popnet_tensor* popnet_empty_like(const popnet_tensor* tensor);

POPNET_API void popnet_free(popnet_tensor* tensor);

POPNET_API popnet_type popnet_dtypeof(const popnet_tensor* tensor);
POPNET_API popnet_shape popnet_shapeof(const popnet_tensor* tensor);
POPNET_API popnet_op popnet_opof(const popnet_tensor* tensor);

POPNET_API bool popnet_set_data(const popnet_tensor* tensor, const void* src, popnet_type src_dtype, size_t offset,
                                size_t n);
POPNET_API bool popnet_get_data(const popnet_tensor* tensor, void* dst, popnet_type dst_dtype, size_t offset, size_t n);

///
/// Graph
///

typedef struct popnet_graph popnet_graph;

POPNET_API popnet_graph* popnet_graph_alloc();
POPNET_API void popnet_graph_free(popnet_graph* graph);

POPNET_API void popnet_graph_build_forward(popnet_graph* graph, popnet_tensor* tensor);
POPNET_API void popnet_graph_compute_forward(popnet_graph* graph);

///
/// Operations
///

POPNET_API popnet_tensor* popnet_add(popnet_tensor* a, popnet_tensor* b);
POPNET_API popnet_tensor* popnet_sub(popnet_tensor* a, popnet_tensor* b);
POPNET_API popnet_tensor* popnet_mul(popnet_tensor* a, popnet_tensor* b);
POPNET_API popnet_tensor* popnet_div(popnet_tensor* a, popnet_tensor* b);

POPNET_API popnet_tensor* popnet_abs(popnet_tensor* a);
POPNET_API popnet_tensor* popnet_neg(popnet_tensor* a);

POPNET_API popnet_tensor* popnet_sqr(popnet_tensor* a);
POPNET_API popnet_tensor* popnet_sqrt(popnet_tensor* a);

POPNET_API popnet_tensor* popnet_exp(popnet_tensor* a);
POPNET_API popnet_tensor* popnet_log(popnet_tensor* a);

POPNET_API popnet_tensor* popnet_sin(popnet_tensor* a);
POPNET_API popnet_tensor* popnet_cos(popnet_tensor* a);
POPNET_API popnet_tensor* popnet_tanh(popnet_tensor* a);

POPNET_API popnet_tensor* popnet_relu(popnet_tensor* a);
POPNET_API popnet_tensor* popnet_sigmoid(popnet_tensor* a);

POPNET_API popnet_tensor* popnet_reduce_sum(popnet_tensor* a, uint8_t axis_mask);
POPNET_API popnet_tensor* popnet_reduce_mean(popnet_tensor* a, uint8_t axis_mask);
POPNET_API popnet_tensor* popnet_reduce_product(popnet_tensor* a, uint8_t axis_mask);
POPNET_API popnet_tensor* popnet_reduce_min(popnet_tensor* a, uint8_t axis_mask);
POPNET_API popnet_tensor* popnet_reduce_max(popnet_tensor* a, uint8_t axis_mask);

POPNET_API popnet_tensor* popnet_matmul(popnet_tensor* a, popnet_tensor* b);
POPNET_API popnet_tensor* popnet_transpose(popnet_tensor* a);

#ifdef __cplusplus
}
#endif

#endif
