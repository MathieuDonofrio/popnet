---
id: REQ-0009
title: Enable CPU architecture optimizations
intents:
  - INT-0002
specifications: []
---

## Obligation

The build shall use `POPNET_CPU_TARGET` to select the compiler target for host-side library code, including CPU kernels. `DEFAULT` shall preserve the toolchain target and be the default. `NATIVE` shall fail during cross-compilation. When the compiler has a native target option, `NATIVE` shall use it. On MSVC x86 and x64, `NATIVE` shall select the highest level among `AVX512`, `AVX2`, and `AVX` that the compiler and host support, or keep the compiler default when no level qualifies. On MSVC ARM64, `NATIVE` shall keep the compiler default. For explicit targets, the build shall use the requested compiler target or fail during configuration if the compiler does not support it.

`POPNET_ENABLE_SIMD` shall default to `ON`. When `OFF`, Popnet shall use its scalar implementation with the same compiler target and compiler auto-vectorization settings.

`POPNET_L1_CACHE_KB` shall override the L1 data cache size independently of the CPU target and SIMD settings. Without an override, only a `NATIVE` build shall probe the host cache, and the build shall use a 32 KiB budget when no size is available.
