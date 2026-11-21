# Popnet

[![Build & Tests](https://github.com/MathieuDonofrio/popnet/actions/workflows/ci.yml/badge.svg)](https://github.com/MathieuDonofrio/popnet/actions/workflows/ci.yml)
[![Code Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMathieuDonofrio%2Fpopnet%2Fcode-metrics-history%2Fcoverage.json)](https://mathieudonofrio.github.io/popnet/#coverage)
[![Requirement Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMathieuDonofrio%2Fpopnet%2Fcode-metrics-history%2Ftraceability.json)](https://mathieudonofrio.github.io/popnet/#traceability)
[![Source Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMathieuDonofrio%2Fpopnet%2Fcode-metrics-history%2Fsloc.json)](https://mathieudonofrio.github.io/popnet/#complexity)
[![P95 Cognitive Complexity](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMathieuDonofrio%2Fpopnet%2Fcode-metrics-history%2Fcognitive-complexity.json)](https://mathieudonofrio.github.io/popnet/#complexity)


A lightweight tensor library for neural networks, with a C API and a C++20 implementation.

## Project Structure

- `include/` — Public C API
- `src/` — Implementation and internal headers
- `src/compute/` — Compute interfaces and backend implementations
- `tests/` — Unit tests
- `bench/` — Benchmarks
- `cmake/` — CMake utilities
- `docs/` — Intents, requirements, specifications and architecture decisions
- `tools/`— Developer tools and their tests

## Supported Platforms

Build & Tests tests three configurations by default. Enable `full_matrix` when running the workflow manually to test all supported combinations below.

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

### C++ tests

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

Coverage is supported on Windows and Linux using the provided CMake presets.

```sh
cmake --preset linux-clang-coverage
cmake --build build/linux-clang-coverage --target coverage
```

On Windows, use `windows-clang-coverage` instead.

Reports are generated as `coverage/index.html`, `coverage/coverage.json` (LLVM JSON), and `coverage/coverage.info` (LCOV) in the corresponding build directory.

## Interactive Code Report

The [interactive code report](https://mathieudonofrio.github.io/popnet/) shows historical trends and the latest published results on `main` for code coverage, requirement coverage, source lines of code, and P95 cognitive complexity. Use the **Branch** selector in Coverage to explore the latest available successful build for an active branch. To browse a run offline, download and extract its `popnet-code-report` artifact and open `index.html` in a browser.

See the [dashboard generator guide](tools/report/README.md) for local generation and metric definitions.

## Benchmarks

Build with `POPNET_BUILD_BENCHMARKS=ON` in Release mode, with sanitizers and coverage disabled.
Run `build/bench/popnet_bench --benchmark_repetitions=5` (add `.exe` on Windows).

## Traceability

The traceability tool navigates and validates relationships between intents, requirements, and tests.

Install dependencies:

```sh
python -m pip install -r tools/trace/requirements.txt
```

Usage:

```sh
python tools/trace/trace.py search "tensor lifetime"
python tools/trace/trace.py show REQ-0012
python tools/trace/trace.py coverage
python tools/trace/trace.py validate
```

All commands support `--json` for structured output. Coverage measures declared requirement-to-test links, not code coverage or verification completeness.
