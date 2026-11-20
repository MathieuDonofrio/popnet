# Code metrics

Install Python 3.10+, install `cloc` with your system package manager, and ensure `cloc` is on PATH. From the repository root, install Lizard and pass the directories you want to measure:

```sh
python -m pip install -r tools/code_metrics/requirements.txt
python tools/code_metrics/code_metrics.py sloc src include
python tools/code_metrics/code_metrics.py cc src include
```

Both commands scan the same C/C++ source and header extensions recursively and count overlapping paths once. Choose source directories that match the intended scope.

SLOC counts physical lines with code, including lines with trailing comments. CC reports the complexity at rank ⌈P × N / 100⌉ in the sorted list of complexities for the N functions recognized by Lizard. It defaults to P95. Use `cc src --percentile 90` to measure a different percentile in one directory.

The default output is one integer. Add `--json` for a structured report with scope and sample counts, or `--badge` for Shields.io endpoint JSON.

Use these measurements to observe changes in codebase size and structure over time. Correctness, clarity, maintainability, and performance take precedence over either value.

Run the tool's tests with both dependencies installed:

```sh
python -m unittest discover -s tools/code_metrics
```
