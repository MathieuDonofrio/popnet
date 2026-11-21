# Traceability

Search and validate links between intents, requirements, specifications, architectural decisions, and tests. Run from the repository root with Python 3.10 or newer:

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
- `coverage [intent-id]` reports declared links and gaps for the whole project or one intent. The percentage is the share of requirements linked to at least one test.
- `validate` checks documents, references, and `// @verifies REQ-NNNN` annotations preceding doctest `TEST_CASE` declarations. Coverage gaps are allowed.

The script defaults to the repository two directories above itself, so it can run from another working directory. Pass `--root <checkout>` before the command to inspect another repository. It requires `docs/intents/` and `docs/requirements/` and scans `docs/specifications/`, `docs/architecture/`, and `tests/` for related artifacts.

Add `--json` for versioned output with repository-relative paths and one-based line numbers. Test records include `comment` when the annotation has a rationale. For Shields.io endpoint JSON, use `coverage --badge` by itself.

Exit status is `0` on success, `1` for an unknown requested ID or failed validation, and `2` for invalid arguments or a repository load failure.

Run the tests with `python -m unittest discover -s tools/trace`.
