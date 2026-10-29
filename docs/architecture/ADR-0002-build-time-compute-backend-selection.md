---
id: ADR-0003
title: Build-time compute backend selection
---

## Context

Different compute backends require different implementations of memory operations and compute kernels.

## Decision

Select the compute backend at build time while keeping the core library independent of backend-specific implementations.

## Rationale

This separates hardware-specific code from common functionality without introducing runtime backend dispatch.

## Consequences

- Each build includes one compute backend.
- Switching backends requires rebuilding the library.
- Backends implement the shared internal compute interfaces.