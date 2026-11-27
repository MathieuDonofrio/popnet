# Popnet

[![Build & Tests](https://github.com/MathieuDonofrio/popnet/actions/workflows/ci.yml/badge.svg)](https://github.com/MathieuDonofrio/popnet/actions/workflows/ci.yml)
[![Code Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMathieuDonofrio%2Fpopnet%2Fcode-metrics-history%2Fcoverage.json)](https://mathieudonofrio.github.io/popnet/#coverage)
[![Requirement Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMathieuDonofrio%2Fpopnet%2Fcode-metrics-history%2Ftraceability.json)](https://mathieudonofrio.github.io/popnet/#traceability)
[![Source Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMathieuDonofrio%2Fpopnet%2Fcode-metrics-history%2Fsloc.json)](https://mathieudonofrio.github.io/popnet/)
[![P95 Cognitive Complexity](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMathieuDonofrio%2Fpopnet%2Fcode-metrics-history%2Fcognitive-complexity.json)](https://mathieudonofrio.github.io/popnet/#complexity)


A lightweight tensor library for neural networks, with a C API and a C++20 implementation.

## Project Structure

- `include/` — Public C API
- `src/` — Implementation and internal headers
- `tests/` — Unit tests
- `bench/` — Benchmarks
- `cmake/` — CMake utilities
- `docs/` — Intents, requirements, specifications and architecture decisions
- `tools/`— Developer tools and their tests

## Supported Platforms

The following matrix is built and tested by the CI.

| Platform | Compilers | Arch |
|---|---|---|
| Windows | MSVC, Clang | x86-64, ARM64 |
| Linux | GCC, Clang | x86-64, ARM64 |
| macOS | Clang | x86-64, ARM64 |

### Compute Backends

- **CPU** - Supported on all platforms.
- **CUDA** - NVIDIA GPUs (not yet implemented).

## Build Requirements

- CMake 3.25+
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

### C++ tests

Configure with tests enabled:

```sh
cmake -S . -B build -G Ninja -DPOPNET_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

For cross-builds, set `CMAKE_CROSSCOMPILING_EMULATOR` to run test discovery and CTest through a compatible emulator. Cross-builds without an emulator can build the test executable for use on compatible hardware.

### Test Coverage

Additional Dependencies:
- Clang or clang-cl
- LLVM coverage tools matching the Clang version

Coverage is supported on Windows and Linux using the provided CMake presets.

```sh
cmake --preset linux-clang-coverage
cmake --build build/linux-clang-coverage --target coverage
```

On Windows, use `windows-clang-coverage` instead.

Reports are generated as `coverage/index.html`, `coverage/coverage.json` (LLVM JSON), and `coverage/coverage.info` (LCOV) in the corresponding build directory.

## Interactive Code Report

The [interactive code report](https://mathieudonofrio.github.io/popnet/) shows metric history for `main` and has a branch selector for the explorers. For offline viewing, extract the `popnet-code-report` artifact from a CI run and open `index.html`. See the [code report guide](tools/report/README.md) for local generation and metric definitions.

## Benchmarks

Build with `POPNET_BUILD_BENCHMARKS=ON` in Release mode, with sanitizers and coverage disabled.
Run `build/bench/popnet_bench --benchmark_repetitions=5` (add `.exe` on Windows).

## Traceability

Use the [traceability tool](tools/trace/README.md) to search and validate declared links between intents, requirements, specifications, and tests.
