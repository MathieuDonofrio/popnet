---
id: ADR-0002
title: Shared SIMD abstractions
---

## Context

Both common CPU code and CPU backend kernels can benefit from SIMD optimizations.

## Decision

Provide shared SIMD abstractions in the common implementation, preferred where applicable while allowing specialized kernels to use SIMD directly.

## Rationale

Shared abstractions allow SIMD functionality to be reused across common code and CPU backend kernels, reducing duplication.

## Consequences

- SIMD abstractions remain independent of compute backends.
- Specialized kernels may use SIMD directly when appropriate.