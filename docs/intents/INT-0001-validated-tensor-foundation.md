---
id: INT-0001
title: Reliable tensor operations for neural networks
---

## Problem

Neural networks need correct tensor operations with clearly documented supported behavior.

## Desired outcome

Provide simple, reliable tensor operations for building neural networks.

## Scope

Tensor allocation, metadata, data transfers, operation tensor creation, graph building, and forward execution, including arithmetic, broadcasting, unary functions, activations, reductions, matrix multiplication, and transpose.

## Success assessment

Verify production optimized operations against naive scalar references, including output shapes and numerical results. Verify tensor lifetime, metadata, data transfers, and graph building and repeated execution. Verify that programmer errors produce debug assertions and unsupported cases are identified clearly.
