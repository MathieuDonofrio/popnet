---
id: ADR-0006
title: Fixed four-dimensional tensor representation
---

## Context

Tensor operations need a common way to represent shape and map coordinates to storage. The shape representation is also part of the public C API.

## Decision

Use exactly four positive dimensions, with singleton dimensions representing lower-rank tensors. Store elements in row-major order with the final dimension varying fastest. The [supported tensor operations specification](../specifications/supported-tensor-operations.md#dimensions) defines the shape and layout conventions.

## Rationale

A fixed rank gives the public API and kernels a uniform shape representation and a known set of axes. The fixed layout gives kernels a contiguous innermost dimension for traversal and vectorization. A dynamically ranked representation would support more ranks and require shape handling that accounts for a variable number of axes.

## Consequences

Callers supply four dimension sizes even for scalars and vectors. Tensors with more than four axes require callers to map them to four dimensions if the operation permits it. Changing the rank would affect the layout of the public shape type and code specialized for four dimensions.
