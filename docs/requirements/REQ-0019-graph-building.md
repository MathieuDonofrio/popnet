---
id: REQ-0019
title: Graph building
intents:
  - INT-0001
specifications: []
---

## Obligation

Popnet shall add an output and its dependencies to a graph without computing values or duplicating tensors.

It shall return true when all are included, or false for a null output or allocation failure. Tensors already added shall remain in the graph.
