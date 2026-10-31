# Supported Tensor Operations

## Dimensions

Tensors have four positive dimensions, written `(d0,d1,d2,d3)`. The last dimension varies fastest. Dimensions of size 1 are singleton axes; `(1,1,1,1)` represents a scalar.

## Unary Operations

Each operation applies elementwise and preserves the input shape.

| Operation | Function | Data Types |
|---|---|---|
| abs | `abs(x)` | f32 |
| neg | `-x` | f32 |
| sqr | `x*x` | f32 |
| sqrt | `sqrt(x)` | f32 |
| exp | `exp(x)` | f32 |
| log | `log(x)` | f32 |
| sin | `sin(x)` | f32 |
| cos | `cos(x)` | f32 |
| tanh | `tanh(x)` | f32 |
| relu | `max(x,0)` | f32 |
| sigmoid | `1/(1+exp(-x))` | f32 |

Trigonometric functions use radians, and `log` is the natural logarithm.

## Binary Operations

Each operation applies elementwise, with `x` from the first operand and `y` from the second.

| Operation | Function | Data Types |
|---|---|---|
| add | `x+y` | f32 |
| sub | `x-y` | f32 |
| mul | `x*y` | f32 |
| div | `x/y` | f32 |

Broadcasting is supported when corresponding dimension sizes match or either size is 1. The output uses the larger size in each dimension, broadcasting singleton axes as needed.

## Reductions

Reductions operate along axes selected by a nonzero 4-bit mask, where bit `i` selects axis `i`.

| Operation | Function | Data Types |
|---|---|---|
| sum | Sum of selected elements | f32 |
| mean | Sum divided by the number of reduced elements | f32 |
| prod | Product of selected elements | f32 |
| min | Minimum of selected elements | f32 |
| max | Maximum of selected elements | f32 |

Selected dimensions become size 1; unselected dimensions retain their sizes.

Selected non-singleton axes must be consecutive when singleton dimensions are ignored. Reductions over only singleton axes are also supported and preserve the input shape and values, including NaNs, infinities, and signed zero.

## Matrix Multiplication

For `A` shaped `(a,b,m,k)` and `B` shaped `(1,1,n,k)`, the output `C` has shape `(a,b,m,n)`:

`C[p,q,i,j] = sum(A[p,q,i,t] * B[0,0,j,t])` for `t = 0..k-1`.

This multiplies `A` by the transpose of `B` over the last two dimensions, broadcasting `B` across the outer dimensions.

## Transpose

For input shape `(a,b,m,n)`, the output shape is `(a,b,n,m)`:

`output[p,q,j,i] = input[p,q,i,j]`