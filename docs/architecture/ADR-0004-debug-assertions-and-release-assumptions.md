---
id: ADR-0004
title: Debug assertions and release assumptions
---

## Context

Callers must satisfy public API preconditions such as compatible operand shapes and in-bounds transfer ranges. Operations can also encounter internal execution paths outside their supported scope, such as a single reduction over separated groups of axes.

## Decision

Use the C `assert` macro for programmer-error checks in debug builds. With `NDEBUG` defined, express checked conditions as compiler assumptions where the compiler provides that facility.

Mark unsupported internal paths unreachable in release builds and assert with a case-specific diagnostic in debug builds. An unsupported operation can satisfy basic argument preconditions while exceeding the documented execution capabilities.

## Rationale

Assertions identify failing checks during development. Compiler assumptions let optimized builds avoid repeated runtime validation of conditions that callers must satisfy.

## Consequences

Failing a debug assertion terminates execution. Violating an assumed precondition or reaching a path marked unreachable can cause undefined behavior in release builds. Recoverable operational failures, such as allocation failure, follow the return-value contract of the API function instead.
