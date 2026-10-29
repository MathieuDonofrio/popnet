---
name: traceability
description: Search, navigate, measure, and validate traceability between intents, requirements, specifications, ADRs, and tests.
---

# Traceability

Use `tools/trace/trace.py` to navigate explicit relationships and check their integrity. The CLI scans the repository on each invocation; it does not modify files. Requires Python 3.10+ and PyYAML (`python -m pip install PyYAML`).

## Commands

Run from the repository root (the script also locates the root when invoked from another directory):

```sh
python tools/trace/trace.py search "tensor lifetime"
python tools/trace/trace.py show REQ-0012
python tools/trace/trace.py coverage
python tools/trace/trace.py coverage INT-0001
python tools/trace/trace.py validate
```

- **`search <query>`** finds intents, requirements, specifications, and ADRs by ID, title, path, or content. Results contain IDs when applicable, titles, and repository-relative paths. Use when the relevant ID is unknown.
- **`show <id>`** resolves an `INT-NNNN`, `REQ-NNNN`, or `ADR-NNNN`. Intents show linked requirements and their test cases; requirements show their intents, specifications, and test cases; ADRs show identity and location. Test results include file paths, line numbers, and names. Use before changing an artifact or its related behavior.
- **`coverage [intent-id]`** reports intent → requirement → test links and missing relationships. Without an ID it covers the project; with `INT-NNNN` it covers that intent's requirements. The percentage is requirements with at least one annotated test divided by total requirements in scope. It is **traceability coverage**, not code coverage.
- **`validate`** checks required front matter, metadata and sections, IDs, filenames, references, and test annotations. Every `TEST_CASE` must have at least one `// @verifies REQ-NNNN` immediately before it; `SUBCASE`s inherit those links. Invalid or duplicate links and unannotated test cases are errors. Missing intent → requirement or requirement → test links are coverage gaps, not validation errors.

## Machine-readable output

Append `--json` to any command for structured output:

```sh
python tools/trace/trace.py show REQ-0012 --json
python tools/trace/trace.py validate --json
```

JSON contains `schema_version` and command-specific fields: `results` for search; artifact identity and relationships for show; `summary`, `intents`, and gaps for coverage; `errors`, `warnings`, and `diagnostics` for validate. Paths are repository-relative; source line numbers are 1-based. Prefer JSON when consuming results programmatically.

For a Shields.io endpoint badge, use **`coverage --badge`** instead of `--json`:

```sh
python tools/trace/trace.py coverage --badge
```

The badge output contains `schemaVersion`, `label`, `message`, and `color`. `--badge` and `--json` cannot be combined.

Exit codes: `0` success (including coverage gaps, validation warnings, or empty search results); `1` unknown requested artifact or validation errors; `2` invalid arguments or failure to load the repository.

## Agent workflow

1. Use `search` if the relevant IDs are unknown, then `show` to follow declared relationships.
2. Read the referenced documents and tests before drawing conclusions or making changes.
3. Use `coverage` to identify missing links; do not assume a gap means missing verification.
4. Run `validate` after editing documents or test annotations. Fix structural errors and broken references without inventing relationships.

The CLI verifies structure and explicit links, **not semantic correctness**. Review consistency between requirements, specifications, tests, and implementation separately, with file and line evidence. An `@verifies` annotation does not establish complete verification of a requirement.
