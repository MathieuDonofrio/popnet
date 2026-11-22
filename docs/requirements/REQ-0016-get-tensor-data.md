---
id: REQ-0016
title: Get tensor data
intents:
  - INT-0001
specifications:
  - docs/specifications/supported-tensor-operations.md#tensor-data-transfers
---

## Obligation

For the specified data type combinations, getting tensor data shall copy the requested element count from the specified tensor element offset to the start of the destination buffer. The source tensor and elements outside the destination range shall remain unchanged, and the function shall return no status.
