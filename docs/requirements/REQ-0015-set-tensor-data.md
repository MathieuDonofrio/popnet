---
id: REQ-0015
title: Set tensor data
intents:
  - INT-0001
specifications:
  - docs/specifications/supported-tensor-operations.md#tensor-data-transfers
---

## Obligation

For the specified data type combinations, setting tensor data shall copy the requested element count from the source buffer into the tensor at the specified element offset. Elements outside the destination range shall remain unchanged, and the function shall return no status.
