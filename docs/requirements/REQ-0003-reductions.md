---
id: REQ-0003
title: Tensor reductions
intents:
  - INT-0001
specifications:
  - docs/specifications/supported-tensor-operations.md
---

## Obligation

Popnet shall compute each specified reduction for any nonempty selection of the tensor's axes, preserving unselected dimensions and setting selected dimensions to 1.
