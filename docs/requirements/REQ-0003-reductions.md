---
id: REQ-0003
title: Tensor reductions
intents:
  - INT-0001
specifications:
  - docs/specifications/supported-tensor-operations.md
---

## Obligation

Popnet shall support each specified reduction over a nonempty selection of axes, provided the selected non-singleton axes are contiguous when singleton dimensions are ignored. Reductions over only singleton axes shall also be supported.
