
# Project

Neural network library

# Project structure

All paths are relative to the repository root.

- `include/`: Public API
- `src/`: Implementation and internal headers
- `src/compute/`: Compute interfaces and backend implementations
- `tests/`: Tests
- `bench/`: Benchmarks
- `cmake/`: Build configuration helpers
- `docs/intents/`: Problems and desired outcomes
- `docs/requirements/`: Required behavior
- `docs/specifications/`: Detailed behavior
- `docs/architecture/`: Design and structure

# Workflow

When changing behavior, review and update the relevant requirements and specifications in the same task.

# Coding style

Follow `.clang-format` and nearby code. Keep related statements together and separate logical steps with blank lines.
In tests, separate setup, execution, checks, and cleanup.

# Build

- Built with CMake
- Configure presets are defined in `CMakePresets.json`

# Tests

- Tests use doctest and are registered with CTest
