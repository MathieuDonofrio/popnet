---
id: ADR-0005
title: Deferred execution with explicit computation graphs
---

## Context

Forward computations can use the same sequence of tensor operations with different input values. The API needs a way to express the dependencies between those operations and control when they run.

## Decision

Operation construction allocates a result tensor and records its operation and input dependencies without computing values. Callers explicitly build a graph from the desired outputs and run a forward pass that computes operations in dependency order.

## Rationale

Recording dependencies separately from execution lets callers build the graph once and rerun it with updated inputs. Eager execution would produce values immediately as operations are called and require the caller to issue those operations again to recompute results.

## Consequences

Callers must build and execute the graph before consuming computed results. Graph construction adds dependency bookkeeping, and operation construction allocates result storage before computation. Each forward pass recomputes every recorded operation, even when inputs are unchanged.
