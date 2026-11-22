# Supported Tensor Operations

## Dimensions

Tensors have four positive dimensions, written `(d0,d1,d2,d3)`. Storage is row-major, with the final dimension varying fastest. Dimensions of size 1 are singleton axes and can represent lower-rank tensors. A scalar has shape `(1,1,1,1)`.

## Tensor Data Transfers

Transfers between a tensor and a host buffer support f32 paired with f32 in either direction. Transfers copy values without data type conversion.

## Numerical Accuracy

- **Exact:** Equals the mathematical result rounded once to the output datatype (nearest, ties to even). Transpose and singleton-axis reductions preserve every input bit instead.
- **Approximate:** Satisfies the absolute-plus-relative bound below.
- **Accumulation-dependent:** Satisfies a bound based on term count and input magnitudes.

For computed result $\hat y$ and exact reference $y$:

$$
|\hat y-y|\leq\mathrm{atol}+\mathrm{rtol}|y|.
$$

The reference is the mathematical result on stored inputs, evaluated exactly or with reference error accounted for. Numerical bounds apply to finite inputs in the function's real domain whose exact result rounds to a finite output. F32 is IEEE 754 binary32.

### Accumulation bounds

For F32, $\gamma_j=ju/(1-ju)$ with $u=2^{-24}$. In reductions, $n$ is the element count, $S$ is the sum of absolute input values, and $P$ is their product. For each matrix output, $k$ is the number of products and $D$ is the sum of their absolute values.

Bounds require $ju<1$, normal or exact-zero inputs and intermediate/final results, and no overflow or underflow. Otherwise, no finite error bound is specified. Evaluation order and fused multiply-add may vary.

## Unary Operations

Each operation applies elementwise and preserves the input shape.

| Operation | Function | Data Type | Numerical Accuracy |
|---|---|---|---|
| abs | `abs(x)` | f32 | Exact |
| neg | `-x` | f32 | Exact |
| sqr | `x*x` | f32 | Exact |
| sqrt | `sqrt(x)` | f32 | Exact |
| exp | `exp(x)` | f32 | Approximate: `atol=1e-6, rtol=1e-5` |
| log | `log(x)` | f32 | Approximate: `atol=1e-6, rtol=1e-5` |
| sin | `sin(x)` | f32 | Approximate: `atol=1e-6, rtol=1e-5` |
| cos | `cos(x)` | f32 | Approximate: `atol=1e-6, rtol=1e-5` |
| tanh | `tanh(x)` | f32 | Approximate: `atol=1e-6, rtol=1e-5` |
| relu | `max(x,0)` | f32 | Exact |
| sigmoid | `1/(1+exp(-x))` | f32 | Approximate: `atol=1e-6, rtol=1e-5` |

Trigonometric functions use radians, and `log` is the natural logarithm.

## Binary Operations

Each operation applies elementwise, with `x` from the first operand and `y` from the second.

| Operation | Function | Data Type | Numerical Accuracy |
|---|---|---|---|
| add | `x+y` | f32 | Exact |
| sub | `x-y` | f32 | Exact |
| mul | `x*y` | f32 | Exact |
| div | `x/y` | f32 | Exact |

Broadcasting is supported when corresponding dimension sizes match or either size is 1. The output uses the larger size in each dimension, broadcasting singleton axes as needed.

## Reductions

Reductions operate along axes selected by a nonzero 4-bit mask, where bit `i` selects axis `i`.

| Operation | Function | Data Type | Numerical Accuracy |
|---|---|---|---|
| sum | Sum of selected elements | f32 | Accumulation-dependent: absolute error at most $\gamma_{n-1}S$ |
| mean | Sum divided by the number of reduced elements | f32 | Accumulation-dependent: absolute error at most $\gamma_n S/n$ |
| prod | Product of selected elements | f32 | Accumulation-dependent: absolute error at most $\gamma_{n-1}\lvert P\rvert$ |
| min | Minimum of selected elements | f32 | Exact |
| max | Maximum of selected elements | f32 | Exact |

All five operations use Exact when reducing only singleton axes.

Selected dimensions become size 1; unselected dimensions retain their sizes.

Selected non-singleton axes must be consecutive when singleton dimensions are ignored. Reductions over only singleton axes are also supported and preserve the input shape.

## Matrix Multiplication

| Operation | Function | Data Type | Numerical Accuracy |
|---|---|---|---|
| matmul | Sum of pairwise products, as defined below | f32 | Accumulation-dependent: absolute error per output at most $\gamma_{2k}D$ |

For `A` shaped `(a,b,m,k)` and `B` shaped `(1,1,n,k)`, the output `C` has shape `(a,b,m,n)`:

`C[p,q,i,j] = sum(A[p,q,i,t] * B[0,0,j,t])` for `t = 0..k-1`.

This multiplies `A` by the transpose of `B` over the last two dimensions, broadcasting `B` across the outer dimensions.

## Transpose

| Operation | Function | Data Type | Numerical Accuracy |
|---|---|---|---|
| transpose | Rearrange elements, as defined below | f32 | Exact |

For input shape `(a,b,m,n)`, the output shape is `(a,b,n,m)`:

`output[p,q,j,i] = input[p,q,i,j]`
