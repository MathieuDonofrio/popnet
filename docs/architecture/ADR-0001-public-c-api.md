---
id: ADR-0001
title: Public C API with internal C++ implementation
---

## Context

Popnet is intended to be simple to use and suitable for integration from other programming languages.

A public C++ API would expose language-specific types and ABI concerns, making interoperability and binding generation more difficult.

At the same time, the implementation benefits from C++ features such as templates, stronger abstractions, and compile-time specialization.

## Decision

Popnet exposes a C-compatible public API in `include/popnet.h` and implements the library internally in C++20.

## Rationale

A C API provides a simple and widely supported integration boundary.

This makes it easier to:
- call Popnet from C and C-compatible environments,
- write bindings for other languages,
- avoid exposing C++ ABI and implementation details as part of the public interface.

Using C++ internally allows the implementation to use templates, abstractions, and architecture-specific optimizations without making them part of the public API.

## Consequences

The public interface uses C-compatible functions, types, and opaque handles.

Ownership and lifetime must be managed explicitly through the API.

Internal C++ implementation details remain hidden behind the C interface.

Changes to the implementation can remain flexible as long as the public C API contract is preserved.