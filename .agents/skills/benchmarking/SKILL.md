---
name: benchmarking
description: Create, run, and analyze performance benchmarks using Google Benchmark.
---

# Benchmarking

## Workflow

1. Read relevant requirements, specifications, and implementation.
2. Identify representative workloads.
3. Create or update benchmarks using Google Benchmark.
4. Run benchmarks and report results.

## Benchmarks

- Measure one operation or execution path at a time.
- Cover representative workloads and edge cases.
- Keep setup outside the timed loop unless explicitly measured.
- Prevent compiler elimination of measured work.
- Use deterministic inputs.
- Prefer a naive implementation as a baseline when practical, but do not require one.

## Measurement

- Use optimized builds without sanitizers or coverage instrumentation.
- Run multiple repetitions and account for variability.
- Record hardware, compiler, build configuration, and source revision.
- Compare implementations under equivalent conditions.

## Reporting

Report measurements, relative performance when a baseline is available, and relevant limitations.

Benchmarks measure performance, not correctness.