---
id: REQ-0012
title: Caller-owned tensor lifetime
intents:
  - INT-0001
specifications: []
---

## Obligation

Popnet shall let callers explicitly free a tensor and its storage without freeing its inputs.
