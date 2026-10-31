#ifndef POPNET_H
#define POPNET_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

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
#elif defined(__clang__)
#define POPNET_EXPECT(cond) __builtin_assume(cond)
#define POPNET_UNREACHABLE() __builtin_unreachable()
#elif defined(__GNUC__)
#define POPNET_EXPECT(cond) ((cond) ? (void)0 : __builtin_unreachable())
#define POPNET_UNREACHABLE() __builtin_unreachable()
#else
#define POPNET_EXPECT(cond) ((void)0)
#define POPNET_UNREACHABLE() ((void)0)
#endif
#endif

#ifdef __cplusplus
#define POPNET_UNSUPPORTED_STATIC(message) []<bool supported = false>() { static_assert(supported, message); }()
#else
#define POPNET_UNSUPPORTED_STATIC(message) _Static_assert(0, message)
#endif
#define POPNET_UNSUPPORTED(message) POPNET_UNREACHABLE()

#define POPNET_MAX_DIMS 4
#define POPNET_MAX_SRC 8

#ifdef __cplusplus
extern "C"
{
#endif

// Tensor

typedef enum
{
  popnet_dtype_none = 0,
  popnet_dtype_f32,
} popnet_dtype;

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

/** Four positive dimensions in row-major order; the last dimension varies fastest. **/
typedef struct popnet_shape
{
  size_t dim[POPNET_MAX_DIMS];
} popnet_shape;

static const popnet_shape popnet_scalar_shape = {.dim = {1, 1, 1, 1}};

POPNET_INLINE size_t popnet_numel(popnet_shape shape)
{
#ifdef __cplusplus
  static_assert(POPNET_MAX_DIMS == 4);
#else
  _Static_assert(POPNET_MAX_DIMS == 4, "Expected four dimensions");
#endif
  size_t size = shape.dim[0];
  POPNET_EXPECT(shape.dim[1] <= SIZE_MAX / size);
  size *= shape.dim[1];
  POPNET_EXPECT(shape.dim[2] <= SIZE_MAX / size);
  size *= shape.dim[2];
  POPNET_EXPECT(shape.dim[3] <= SIZE_MAX / size);
  size *= shape.dim[3];
  POPNET_EXPECT(size > 0);
  return size;
}

typedef struct popnet_tensor popnet_tensor;

/**
 * Allocates independent, uninitialized storage with no recorded operation.
 * Release the result with popnet_free().
 *
 * @pre dtype is supported and every shape dimension is positive.
 **/
POPNET_API popnet_tensor* popnet_empty(popnet_dtype dtype, popnet_shape shape);
/** Allocates like popnet_empty() using tensor's shape and type; does not copy its data or operation. **/
POPNET_API popnet_tensor* popnet_empty_like(const popnet_tensor* tensor);

/** Frees the tensor and its storage without freeing its input tensors. **/
POPNET_API void popnet_free(popnet_tensor* tensor);

POPNET_API popnet_dtype popnet_dtypeof(const popnet_tensor* tensor);
POPNET_API popnet_shape popnet_shapeof(const popnet_tensor* tensor);
POPNET_API popnet_op popnet_opof(const popnet_tensor* tensor);

/**
 * Copies n elements from src into the tensor at element offset; other elements are unchanged.
 * Currently both source and tensor must use f32.
 *
 * @pre tensor and src are non-null, src holds n elements, and the tensor range is in bounds.
 **/
POPNET_API void popnet_set_data(const popnet_tensor* tensor, const void* src, popnet_dtype src_dtype, size_t offset,
                                size_t n);
/**
 * Copies n elements from the tensor at element offset into dst without changing the tensor.
 * Currently both tensor and destination must use f32.
 *
 * @pre tensor and dst are non-null, dst holds n elements, and the tensor range is in bounds.
 **/
POPNET_API void popnet_get_data(const popnet_tensor* tensor, void* dst, popnet_dtype dst_dtype, size_t offset,
                                size_t n);

// Graph

typedef struct popnet_graph popnet_graph;

/** Allocates an empty graph; release it with popnet_graph_free(). **/
POPNET_API popnet_graph* popnet_graph_alloc();
/** Frees the graph without freeing its tensors. **/
POPNET_API void popnet_graph_free(popnet_graph* graph);

/**
 * Adds the output and its dependencies without computing values or duplicating tensors.
 *
 * The graph borrows these tensors; keep them alive during graph building and computation.
 *
 * If allocation fails, tensors already added remain in the graph.
 *
 * @pre graph is non-null.
 *
 * @return true if the output and all dependencies are included; false if the
 *         output is null or allocation fails.
 **/
POPNET_API bool popnet_graph_build_forward(popnet_graph* graph, popnet_tensor* tensor);
/**
 * Executes each recorded operation once, after its dependencies, using current input values.
 * Tensors with no operation are unchanged. Referenced tensors must remain alive during computation.
 **/
POPNET_API void popnet_graph_compute_forward(popnet_graph* graph);

/**
 * @name Operation creation
 * These functions allocate independent result storage and record inputs without computing values.
 * Release results with popnet_free(); inputs remain caller-owned and must outlive graph use.
 * Computing an operation preserves its inputs. Shapes, types, and dependencies are fixed at creation.
 *
 * @pre Inputs are non-null and use the same supported type (currently f32).
 * @{
 **/

/**
 * @name Elementwise binary operations
 * Corresponding dimensions must match or one must be 1; singleton dimensions broadcast.
 * The output uses the larger size in each dimension. Subtraction and division use operand order a, b.
 * @{
 **/
POPNET_API popnet_tensor* popnet_add(popnet_tensor* a, popnet_tensor* b);
POPNET_API popnet_tensor* popnet_sub(popnet_tensor* a, popnet_tensor* b);
POPNET_API popnet_tensor* popnet_mul(popnet_tensor* a, popnet_tensor* b);
POPNET_API popnet_tensor* popnet_div(popnet_tensor* a, popnet_tensor* b);
/** @} **/

/**
 * @name Elementwise unary operations
 * Preserve the input shape. Angles are in radians; log is the natural logarithm.
 * ReLU is max(x, 0); sigmoid is 1 / (1 + exp(-x)).
 * @{
 **/
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
/** @} **/

/**
 * @name Reductions
 * Bit i of axis_mask selects dimension i; selected output dimensions become 1.
 * Execution supports one contiguous group of reduced dimensions, ignoring singleton dimensions.
 * Compose separate reduction operations for groups separated by retained non-singleton dimensions.
 *
 * @pre axis_mask is nonzero and uses only the lowest POPNET_MAX_DIMS bits.
 * @{
 **/
POPNET_API popnet_tensor* popnet_reduce_sum(popnet_tensor* a, uint8_t axis_mask);
POPNET_API popnet_tensor* popnet_reduce_mean(popnet_tensor* a, uint8_t axis_mask);
POPNET_API popnet_tensor* popnet_reduce_product(popnet_tensor* a, uint8_t axis_mask);
POPNET_API popnet_tensor* popnet_reduce_min(popnet_tensor* a, uint8_t axis_mask);
POPNET_API popnet_tensor* popnet_reduce_max(popnet_tensor* a, uint8_t axis_mask);
/** @} **/

/**
 * Records A * transpose(B), sharing B across the outer batches.
 * The result has shape (a,b,m,n).
 *
 * @pre A has shape (a,b,m,k) and B has shape (1,1,n,k).
 **/
POPNET_API popnet_tensor* popnet_matmul(popnet_tensor* a, popnet_tensor* b);
/** Swaps the final two dimensions independently in each outer batch. **/
POPNET_API popnet_tensor* popnet_transpose(popnet_tensor* a);
/** @} **/

#ifdef __cplusplus
}
#endif

#endif
