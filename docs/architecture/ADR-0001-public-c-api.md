---
id: ADR-0001
title: Public C API with internal C++ implementation
---

## Context

Popnet is intended to be simple to use and suitable for integration from other programming languages.

A public C++ API would expose language-specific types and ABI concerns, making interoperability and binding generation more difficult.

At the same time, the implementation benefits from C++ features such as templates, stronger abstractions, and compile-time specialization.

## Decision

The library exposes a C-compatible public API in `include/popnet.h` and uses modern C++ internally.

## Rationale

A C API provides a widely supported integration boundary for C callers and bindings in other languages. It keeps the C++ ABI and implementation details out of the public interface.

Using C++ internally allows the implementation to use templates, abstractions, and architecture-specific optimizations without making them part of the public API.

## Consequences

The public interface is limited to C-compatible functions and types, including opaque handles. Callers manage ownership and lifetime explicitly through the API. The implementation can evolve behind the public C contract.
