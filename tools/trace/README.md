# Traceability

The traceability tool searches a repository's intents, requirements, specifications, and architectural decisions. It follows declared links to specifications and tests, and validates their structure and references.

Install Python 3.10+ and the tool's dependency. From the repository root, run:

```sh
python -m pip install -r tools/trace/requirements.txt
python tools/trace/trace.py search "lifetime"
python tools/trace/trace.py show REQ-0001
python tools/trace/trace.py coverage
python tools/trace/trace.py coverage INT-0001
python tools/trace/trace.py validate
```

- `search <query>` looks for IDs, titles, paths, and text.
- `show <id>` displays an intent, requirement, or architectural decision and any links reported for that artifact.
- `coverage [intent-id]` reports declared intent-to-requirement and requirement-to-test links for the whole project or one intent. The percentage is the share of requirements with at least one annotated test. Gaps indicate missing links. Use the linked test cases to assess what they actually verify.
- `validate` checks document metadata, filenames, required sections, references, and `// @verifies REQ-NNNN` annotations preceding doctest `TEST_CASE` declarations. Coverage gaps are allowed.

The script locates the repository two directories above itself, so it can run from another working directory. It requires `docs/intents/` and `docs/requirements/` and scans `docs/specifications/`, `docs/architecture/`, and `tests/` for related artifacts.

Add `--json` to any command for structured output with a `schema_version`. Paths are repository-relative, and line numbers start at 1. Use `coverage --badge` to generate Shields.io endpoint JSON. The two output flags are mutually exclusive.

Exit status is `0` on success, `1` for an unknown requested ID or failed validation, and `2` for invalid arguments or a repository load failure.

Run the tool's tests after installing its dependency:

```sh
python -m unittest discover -s tools/trace
```
