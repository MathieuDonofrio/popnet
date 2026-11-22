---
id: REQ-0019
title: Graph building
intents:
  - INT-0001
specifications: []
---

## Obligation

Graph building shall add the existing output tensor and its dependencies without computing values. Each tensor shall be included at most once.

The function shall return true when all are included and false for a null output or allocation failure. Tensors already added shall remain in the graph.
