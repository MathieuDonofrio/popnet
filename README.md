# Popnet

[![CI (build and tests)](https://github.com/MathieuDonofrio/popnet/actions/workflows/ci.yml/badge.svg)](https://github.com/MathieuDonofrio/popnet/actions/workflows/ci.yml)
[![Code coverage](https://codecov.io/gh/MathieuDonofrio/popnet/graph/badge.svg)](https://app.codecov.io/gh/MathieuDonofrio/popnet)

A lightweight tensor library for neural networks, with a C API and a C++20 implementation.

## Project Structure

- `include/` — Public C API
- `src/` — Implementation and internal headers
- `src/compute/` — Compute interfaces and backend implementations
- `tests/` — Unit tests
- `bench/` — Benchmarks
- `cmake/` — CMake utilities
- `docs/` — Intents, requirements, specifications and architecture

## Supported Platforms

The following platforms and compilers are built and tested by the CI/CD.

| Platform | Compilers |
|---|---|
| Windows | MSVC, Clang |
| Linux | GCC, Clang |
| macOS | Clang |

### Compute Backends

- **CPU** — Supported on all platforms.
- **CUDA** — NVIDIA GPUs (not yet implemented).

## Build Requirements

- CMake 3.20+
- C++20-compatible compiler
- Ninja
- Git (for fetching dependencies)

### Dependencies

Automatically fetched by CMake when enabled:

- [doctest](https://github.com/doctest/doctest) v2.4.12 — Testing
- [Google Benchmark](https://github.com/google/benchmark) v1.9.4 — Benchmarking

## Build

Configure and build:

```sh
cmake -S . -B build -G Ninja
cmake --build build
```

## Running Tests

Configure with tests enabled:

```sh
cmake -S . -B build -G Ninja -DPOPNET_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

### Test Coverage

Additional Dependencies:
- Clang or clang-cl
- LLVM coverage tools matching the Clang version
- gcovr 8.5+

Coverage is supported on Windows and Linux using the provided CMake presets.

```sh
cmake --preset linux-clang-coverage
cmake --build build/linux-clang-coverage --target coverage
```

On Windows, use `windows-clang-coverage` instead.

Reports are generated as `coverage.html` and `coverage.xml` in the corresponding build directory.

CI builds and tests share jobs, so they use one combined workflow badge. The coverage badge shows Codecov's measured coverage for the default branch, using the Linux coverage build with native optimization disabled.

#### Codecov setup

1. Sign in to [Codecov](https://app.codecov.io/gh/MathieuDonofrio/popnet) with GitHub and enable this repository, granting the Codecov GitHub App access if prompted.
2. Push the workflow to the default branch and confirm the **Upload to Codecov** step succeeds. The coverage badge remains unknown until Codecov processes the first report.

Uploads use [GitHub OIDC](https://github.com/codecov/codecov-action#using-oidc); no `CODECOV_TOKEN` secret is required. Upload failures fail the coverage job, but its HTML/XML artifact is uploaded first. Fork and Dependabot pull requests still collect coverage and save artifacts, but skip Codecov uploads.

For a private repository, use the badge URL supplied by Codecov under **Configuration → Badges and Graphs** if it requires a badge access token; never put an upload token in the README.
