---
id: ADR-0002
title: Shared SIMD abstractions
---

## Context

Common CPU code and CPU backend kernels can use the same SIMD primitives and traversal patterns. Some kernels need instruction selection or data access patterns specific to the operation.

## Decision

Keep reusable SIMD abstractions in the common implementation, independent of compute backends. Prefer them when they fit the operation. Specialized kernels may use SIMD directly.

## Rationale

Sharing SIMD primitives and traversal logic reduces duplication and lets both users benefit from maintenance in one place. Direct use gives specialized kernels control over instruction selection and data access where the shared abstractions would constrain them.

## Consequences

Shared abstractions need to accommodate common use cases without taking on each kernel's special cases. Kernels that use SIMD directly retain more control and may need their own architecture-specific code and maintenance.
