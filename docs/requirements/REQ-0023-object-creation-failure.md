---
id: REQ-0023
title: Object creation failure
intents:
  - INT-0001
specifications: []
---

## Obligation

If allocation fails while creating a library object, Popnet shall return null and release any resources allocated for that object.
