---
id: ADR-0003
title: Build-time compute backend selection
---

## Context

Different compute backends require different implementations of memory operations and compute kernels.

## Decision

Select one compute backend at build time. The core library calls shared internal compute interfaces implemented by the selected backend.

## Rationale

The shared interfaces separate hardware-specific code from common functionality. Build-time selection avoids runtime dispatch between backends.

## Consequences

A compiled library cannot switch compute backends at runtime. Using a different backend requires a separate build of the library.
