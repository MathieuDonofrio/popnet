---
id: INT-0002
title: Efficient tensor execution across CPUs and GPUs
---

## Problem

Different CPUs and GPUs offer different capabilities for accelerating tensor operations.

## Desired outcome

Run tensor operations efficiently on any CPU or GPU through kernels optimized for the selected hardware.

## Scope

Backend execution and architecture-specific optimization of library code and kernels.

## Success assessment

Confirm that the selected backend enables available architecture-specific optimizations. Assess execution efficiency with benchmarks.
