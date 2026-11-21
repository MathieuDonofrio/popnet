# Code metrics

With Python 3.10 or newer, install Lizard from the repository root and pass the directories you want to measure:

```sh
python -m pip install -r tools/code_metrics/requirements.txt
python tools/code_metrics/code_metrics.py sloc src include
python tools/code_metrics/code_metrics.py cognitive src include
```

Both commands scan the same C/C++ source and header extensions recursively and count overlapping paths once. Choose source directories that match the intended scope.

Source Lines of Code sums Lizard's per-file NLOC (non-comment lines of code), including code outside functions. Lizard's Cognitive Complexity extension gives nested control flow more weight and counts an entire switch as one decision. The tool reports the score at rank ⌈P × N / 100⌉ in the sorted list for the N recognized functions. It defaults to P95. Use `cognitive src --percentile 90` for a different percentile in one directory.

The default output is one integer. Add `--json` for a structured report with scope and sample counts, or `--badge` for Shields.io endpoint JSON. The Cognitive Complexity badge color moves from green at 0 through yellow at 15 to red at 30 and above. The color is display-only.

Use `cognitive src include --json --functions` to add `function_details` to the JSON. Each record includes the Lizard function name, file path, one-based start and end lines, and its cognitive complexity score. Paths within the working directory are relative to it. The percentile and function count use the same records.

Use these measurements to observe changes in codebase size and structure over time. Correctness, clarity, maintainability, and performance take precedence over either value.

Run the tool's tests with Lizard installed:

```sh
python -m unittest discover -s tools/code_metrics
```
