---
id: REQ-0009
title: Enable CPU architecture optimizations
intents:
  - INT-0002
specifications: []
---

## Obligation

When POPNET_NATIVE is enabled, the build shall enable architecture-specific compiler optimizations for host-side library code (CPU code outside the compute kernels).

When POPNET_NATIVE is enabled and the CPU backend is selected, the build shall also enable architecture-specific compiler optimizations for CPU kernels.

Selected instructions shall be supported by the host CPU and operating system.
