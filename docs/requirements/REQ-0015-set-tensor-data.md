---
id: REQ-0015
title: Set tensor data
intents:
  - INT-0001
specifications: []
---

## Obligation

Popnet shall copy the requested element count from a source buffer into the tensor at the specified element offset, interpreting the buffer using its declared type and converting to the tensor type where supported. Elements outside the requested range shall remain unchanged. The operation shall return no status.
