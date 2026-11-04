---
name: testing
description: Write, update, and review doctest tests when verifying required behavior, boundary cases, and failure conditions.
---

# Testing

## Workflow

1. Read relevant requirements, specifications, and existing tests.
2. Derive expected behavior from requirements and specifications, never from the implementation. If the contract is unclear, report the ambiguity rather than guessing.
3. Add or update focused tests, matching nearby test style.
4. Apply the quality criteria and run the affected tests.

## Annotation

Every `TEST_CASE` must have at least one valid `// @verifies REQ-NNNN` annotation
immediately preceding it, one line per requirement. The rationale is optional.
`SUBCASE`s inherit their parent `TEST_CASE` annotations.

```cpp
// @verifies REQ-NNNN: <optional rationale>
TEST_CASE("<behavior being tested>")
```

Link only existing requirement IDs selected by the behavior the test actually
verifies. If no existing requirement meaningfully applies, report the gap rather
than leaving the test unannotated or inventing an association. A link does not
imply complete coverage.

## Quality

- Focused: Each test case checks one behavior; multiple requirements are linked only when coupled in that behavior.
- Meaningful: Assertions check required outcomes, not merely that code executes.
- Independent: Expected results do not repeat the implementation being tested.
- Representative: Cases include relevant boundaries and failure conditions without redundant variations.
- Repeatable: Tests produce reproducible results and do not depend on execution order.
- Maintainable: Setup is minimal and failures identify the behavior that broke.

## Architecture independence

Verify observable behavior with the same architecture-independent inputs and
expected results across scalar and SIMD implementations. Do not conditionally
compile test cases or assertions using macros. Rely on the build configuration and CI matrix to select the implementation.

## Verification

Run affected tests through CTest using the project's CMake presets.
Check requirement references and report failures or verification gaps.
