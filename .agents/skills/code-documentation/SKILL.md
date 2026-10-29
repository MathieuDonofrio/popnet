---
name: code-documentation
description: Create, update, and review in-code documentation for public APIs and internal implementations.
---

# Code Documentation

## Workflow

1. Read the relevant requirements, specifications, and existing code documentation.
2. Identify information necessary to understand or correctly use the code that is not apparent from the code itself.
3. Add or update documentation following nearby conventions.
4. Review documentation for accuracy, clarity, consistency, and redundancy.
5. Report unclear or conflicting contracts rather than inventing behavior.

## Public API

Use `/** ... **/` Doxygen block comments for public API documentation, with leading
`*` on multiline comments. Separate paragraphs and tag groups with a blank comment
line. Keep short comments on one line; add tags only when they clarify the contract.

Prefer `@return` for simple results, including booleans. Use `@retval` when distinct
values need individual explanations. Describe state changes on failure in prose.

Document relevant aspects of the API contract, including:
- Purpose and semantics
- Preconditions and parameter constraints
- Ownership and lifetime
- Return values and failure behavior
- Side effects and other non-obvious guarantees

Document shared contracts once where practical. Avoid repeating detailed specifications.

## Internal Implementation

Document non-obvious:
- Algorithms and design decisions
- Invariants and assumptions
- Performance optimizations and tradeoffs
- Platform-specific behavior
- Safety and correctness constraints

Prefer explaining **why** something is implemented a particular way rather than describing what individual statements do.

## Quality

- Accurate: Documentation reflects the defined contract and implementation.
- Clear: Language is concise, precise, and unambiguous.
- Relevant: Comments provide information not readily apparent from the code.
- Consistent: Documentation agrees with requirements and specifications.
- Maintainable: Avoid unnecessary duplication and details likely to become outdated.

## Verification

Review affected documentation against requirements, specifications, and implementation.

Do not change behavior solely to satisfy documentation. Report inconsistencies that require a separate decision.
