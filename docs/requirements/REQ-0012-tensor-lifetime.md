---
id: REQ-0012
title: Caller-owned tensor lifetime
intents:
  - INT-0001
specifications: []
---

## Obligation

Callers shall be able to explicitly free a tensor and its storage without freeing its inputs.
