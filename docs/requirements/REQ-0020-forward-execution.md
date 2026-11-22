---
id: REQ-0020
title: Compute a forward graph
intents:
  - INT-0001
specifications: []
---

## Obligation

Each forward graph execution shall compute every recorded operation once, after its dependencies, using the current input values. Tensors with no recorded operation shall remain unchanged.
