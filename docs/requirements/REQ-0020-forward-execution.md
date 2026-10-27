---
id: REQ-0020
title: Compute a forward graph
intents:
  - INT-0001
specifications: []
---

## Obligation

Each time a graph is computed, Popnet shall execute every operation once, after its dependencies, using the current input values. Tensors with no recorded operation shall remain unchanged.
