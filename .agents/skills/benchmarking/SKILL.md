---
name: benchmarking
description: Create, run, and analyze Google Benchmark benchmarks when evaluating optimizations or checking performance regressions between revisions.
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
- Compare against a naive reference when practical to assess an optimization; this does not establish performance changes between revisions.

## Measurement

- Use optimized builds without sanitizers or coverage instrumentation.
- Run multiple repetitions and account for variability.
- Record hardware, compiler, build configuration, and source revision.
- Compare implementations under equivalent hardware, compiler settings, and build configurations.

## Performance regressions

For before/after comparisons, measure a baseline using the original implementation,
then benchmark the modified implementation with the same representative workloads
and equivalent measurement conditions described above. Use repeated measurements
for both revisions and account for variability before attributing differences to
the change. Investigate reproducible regressions.

## Reporting

Report measurements, measured differences relative to an available baseline, and relevant limitations. If the original implementation cannot be measured, explicitly report the missing baseline rather than claiming performance equivalence.

Benchmarks measure performance, not correctness.