---
id: REQ-0018
title: Graph lifetime
intents:
  - INT-0001
specifications: []
---

## Obligation

Popnet shall let callers allocate an empty graph and free it without freeing its tensors.

Callers must keep referenced tensors alive during graph building and computation.
