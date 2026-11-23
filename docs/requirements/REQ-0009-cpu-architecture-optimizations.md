---
id: REQ-0009
title: Enable CPU architecture optimizations
intents:
  - INT-0002
specifications: []
---

## Obligation

The build shall allow an explicit `POPNET_SIMD` selection for host-side library code, including CPU kernels. An explicit selection shall override native detection and fail configuration if the compiler target does not support it. `SCALAR` shall select the library's scalar path.

`POPNET_SIMD` shall default to `AUTO`, which selects optimizations for the build host. Cross-builds with `AUTO` shall use the compiler target defaults.
