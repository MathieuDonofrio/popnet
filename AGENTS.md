
# Project

Neural network library

# Project structure

All paths are relative to the repository root.

- `include/`: Public API
- `src/`: Implementation and internal headers
- `src/compute/`: Compute interfaces and backend implementations
- `tests/`: Tests
- `bench/`: Benchmarks
- `tools/`: Developer tools and their tests
- `cmake/`: Build configuration helpers
- `docs/intents/`: Problems and desired outcomes
- `docs/requirements/`: Required behavior
- `docs/specifications/`: Detailed behavior
- `docs/architecture/`: Architectural decision records

# Workflow

When changing any of the following, review the related artifacts and ensure they remain consistent:
- Implementation and behavior
- Intents, requirements, specifications, and architecture
- Tests
- In-code documentation
- Project documentation (README.md, AGENT.md, skills, etc)
Update affected artifacts as needed.

If requirements or specifications are unclear or conflicting, report the ambiguity rather than making assumptions.

# Coding style

Follow `.clang-format` and nearby code. Keep related statements together and separate logical steps with blank lines.
In tests, separate setup, execution, checks, and cleanup.

# Build

- Built with CMake
- Configure presets are defined in `CMakePresets.json`

# Tests

- Tests use doctest and are registered with CTest
