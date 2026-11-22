---
id: INT-0001
title: Reliable tensor operations for neural networks
---

## Problem

Unclear tensor behavior and numerical errors make neural-network results difficult to diagnose.

## Desired outcome

Developers can build neural networks using reliable tensor computations with predictable behavior, well-defined ownership, and numerically correct results.

## Scope

Tensor allocation, metadata, data transfers, operation tensor creation, graph building, and forward execution, including arithmetic, broadcasting, unary functions, activations, reductions, matrix multiplication, and transpose.

## Success assessment

Tensor lifetime, metadata, transfers, and graph construction and reuse meet their functional contracts. Output shapes match reference implementations, and computed values agree within the specified numerical bounds.
