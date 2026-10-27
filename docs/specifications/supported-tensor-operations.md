# Supported tensor operations

## Dimensions

Shapes have four positive dimensions, written `(d0,d1,d2,d3)`. The last dimension varies fastest. Dimensions of size 1 represent singleton axes; `(1,1,1,1)` represents a scalar.

## Unary operations

The output preserves the input shape and applies the selected function to each element.

| Operation | Function |
|---|---|
| abs | absolute value |
| neg | `-x` |
| sqr | `x*x` |
| sqrt | square root |
| exp | natural exponential |
| log | natural logarithm |
| sin | sine in radians |
| cos | cosine in radians |
| tanh | hyperbolic tangent |
| relu | `max(x,0)` |
| sigmoid | `1/(1+exp(-x))` |

## Binary operations

Add, subtract, multiply, and divide apply elementwise in operand order. In each dimension, operand sizes must match or one must be 1. The output uses the larger size, and singleton axes broadcast across that dimension.

## Reductions

Sum, mean, product, minimum, and maximum aggregate elements along any nonempty selection of axes identified by an axis mask. Selected dimensions become 1; other dimensions retain their sizes. Mean divides the sum by the number of aggregated elements.

## Matrix multiplication

For `A` shaped `(a,b,m,k)` and `B` shaped `(1,1,n,k)`, the output `C` has shape `(a,b,m,n)`:

`C[p,q,i,j] = sum over t=0..k-1 of A[p,q,i,t] * B[0,0,j,t]`.

This computes `A * transpose(B)`, sharing the right operand across the outer batches.

## Transpose

For input shape `(a,b,m,n)`, the output shape is `(a,b,n,m)` and `output[p,q,j,i] = input[p,q,i,j]`.
