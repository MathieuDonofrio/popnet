---
name: test-coverage
description: Collect and inspect C++ code coverage using the project's CMake, CTest, and LLVM coverage workflow.
---

# Test coverage

## Collect

Select `windows-clang-coverage` or `linux-clang-coverage` for the current OS:

```sh
cmake --preset <preset>
cmake --build build/<preset> --target coverage
```

Windows requires an x64 Visual Studio developer environment and a compatible
Clang installation (the current MSVC standard library requires Clang 19+).
The target requires matching LLVM profile and coverage tools.
Report missing tools; do not install dependencies as part of this skill.

The target clears old profiles, runs the full CTest suite, and writes
`coverage/coverage.info` (LCOV) and `coverage/index.html` with annotated source pages in the build
directory. Confirm the command succeeded and the reports contain library files.

## Inspect

Use the LCOV, HTML, and source files to examine overall line and branch coverage,
relevant files, uncovered paths, and partially covered conditions. Include file
paths and line numbers for specific gaps. Identify filtered runs explicitly.

Coverage records execution, not correctness. Uncovered code does not necessarily
contain defects, and high coverage does not prove sufficient testing. Account
for platform and compile-time exclusions; these presets use the portable CPU
path with native optimization disabled.

Expand template instantiations in the LLVM HTML report to distinguish executed
specializations from discarded code. Compile-time conditions are not runtime
branches; absent native SIMD paths are not measured by the portable preset.

Compare runs with the same source revision, platform, configuration, scope, and
test selection where possible. Summarize the platform, test status, relevant
measurements, gaps, limitations, and report paths. Never invent results.

This skill collects and analyzes coverage. Do not modify production code or
tests, add thresholds, or maintain a separate coverage database.
