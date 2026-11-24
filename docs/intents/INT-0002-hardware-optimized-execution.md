---
id: INT-0002
title: Efficient tensor execution across CPUs and GPUs
---

## Problem

Different CPUs and GPUs offer different capabilities for accelerating tensor operations. A build optimized for one CPU may require instructions unavailable on another.

## Desired outcome

Run tensor operations efficiently on supported hardware architectures. Let builders choose the CPU compatibility they need and compare explicit SIMD against compiler-generated code under the same target.

## Scope

Backend execution and architecture-specific optimization of library code and kernels.

## Success assessment

Verify that portable and specialized builds can be configured and that they execute correctly on compatible hardware. Assess whether architecture-specific optimizations improve execution efficiency by benchmarking representative operations and tensor shapes against comparable baselines.
