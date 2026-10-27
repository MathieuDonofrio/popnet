---
name: testing
description: Write, update, and review project tests.
---

# Testing

## Workflow

1. Read relevant requirements, specifications, and existing tests.
2. Identify the behavior to verify and derive expected results from its contract.
3. Add or update focused tests, matching nearby test style.
4. Apply the quality criteria and run the affected tests.

## Annotation

Annotate each TEST_CASE as shown below, one line per requirement. The rationale
is optional, and annotations include the test's subcases.

```cpp
// @verifies REQ-NNNN: <optional rationale>
TEST_CASE("<behavior being tested>")
```

Link only existing requirements the test actually checks. Report gaps rather
than invent links. A link does not imply complete coverage.

## Quality

- Focused: Each test case checks one behavior; multiple requirements are linked only when coupled in that behavior.
- Meaningful: Assertions check required outcomes, not merely that code executes.
- Independent: Expected results do not repeat the implementation being tested.
- Representative: Cases include relevant boundaries and failure conditions without redundant variations.
- Repeatable: Tests produce reproducible results and do not depend on execution order.
- Maintainable: Setup is minimal and failures identify the behavior that broke.

## Verification

Run affected tests through CTest using the project's CMake presets.
Check requirement references and report failures or verification gaps.
