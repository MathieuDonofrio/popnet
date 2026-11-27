---
id: REQ-0003
title: Tensor reductions
intents:
  - INT-0001
specifications:
  - docs/specifications/supported-tensor-operations.md#reductions
---

## Obligation

Each specified reduction shall support nonempty selections of axes whose non-singleton axes are contiguous when singleton dimensions are ignored, including selections of only singleton axes.
