---
id: REQ-0009
title: Enable CPU architecture optimizations
intents:
  - INT-0002
specifications: []
---

## Obligation

When `POPNET_NATIVE` is enabled, the build shall enable architecture-specific compiler optimizations for host-side library code outside the compute kernels and, when the CPU backend is selected, for CPU kernels.
