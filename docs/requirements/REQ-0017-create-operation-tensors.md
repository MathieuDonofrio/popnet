---
id: REQ-0017
title: Create operation tensors
intents:
  - INT-0001
specifications: []
---

## Obligation

Operation creation shall allocate a result tensor with independent storage, record the operation and its inputs, and set the shape and data type according to the operation. Creation shall not compute values.
