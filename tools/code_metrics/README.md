# Code metrics

Install Python 3.10+, install `cloc` with your system package manager, and ensure `cloc` is on PATH. From the repository root, install Lizard and pass the directories you want to measure:

```sh
python -m pip install -r tools/code_metrics/requirements.txt
python tools/code_metrics/code_metrics.py sloc src include
python tools/code_metrics/code_metrics.py cognitive src include
```

Both commands scan the same C/C++ source and header extensions recursively and count overlapping paths once. Choose source directories that match the intended scope.

Source Lines of Code counts physical lines with code, including lines with trailing comments. Lizard's Cognitive Complexity extension gives nested control flow more weight and counts an entire switch as one decision. The tool reports the score at rank ⌈P × N / 100⌉ in the sorted list for the N recognized functions. It defaults to P95. Use `cognitive src --percentile 90` for a different percentile in one directory.

The default output is one integer. Add `--json` for a structured report with scope and sample counts, or `--badge` for Shields.io endpoint JSON. The Cognitive Complexity badge color moves from green at 0 through yellow at 15 to red at 30 and above. The color is display-only.

Use these measurements to observe changes in codebase size and structure over time. Correctness, clarity, maintainability, and performance take precedence over either value.

Run the tool's tests with both dependencies installed:

```sh
python -m unittest discover -s tools/code_metrics
```
