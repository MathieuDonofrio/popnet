---
id: ADR-0007
title: Packed ternary storage in the CPU backend
---

## Context

The CPU backend needs a compact ternary layout that supports word and SIMD processing.

## Decision

The CPU backend packs each row into consecutive blocks. Each block stores its nonzero bitplane followed by its negative bitplane. The bit pairs for -1, 0, and +1 are (1,1), (0,0), and (1,0), respectively. Within each plane, elements fill bytes from the least-significant bit upward.

A full plane matches the SIMD register width or the 64-bit word size used by scalar builds. In a partial block, each plane uses only enough bytes to hold its elements. Packing clears unused bits. Rows are byte-aligned and consecutive.

## Rationale

Full planes fit a scalar word or SIMD register. Compact tails keep short and irregular rows small.

## Consequences

The byte layout can vary with the CPU target. Other backends choose their own storage layout.
