---
id: REQ-0016
title: Get tensor data
intents:
  - INT-0001
specifications: []
---

## Obligation

Popnet shall copy the requested element count from the specified tensor element offset into a destination buffer, converting from the tensor type to the declared destination type where supported. The tensor shall remain unchanged. The operation shall return no status.
