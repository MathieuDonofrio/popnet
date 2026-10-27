---
id: REQ-0021
title: Programmer error assertions
intents:
  - INT-0001
specifications: []
---

## Obligation

In debug builds, Popnet shall assert API preconditions, including valid shapes, in-bounds data transfers, and compatible operands. In release builds, the compiler shall be allowed to assume those conditions are true.
