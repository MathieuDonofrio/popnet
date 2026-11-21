# Interactive code report

From the repository root, with Python 3.10 or newer:

```sh
cmake --preset linux-clang-coverage
cmake --build build/linux-clang-coverage --target coverage
python -m pip install -r tools/trace/requirements.txt -r tools/code_metrics/requirements.txt -r tools/report/requirements.txt
python tools/report/generate.py --coverage-json build/linux-clang-coverage/coverage/coverage.json
```

On Windows, use an x64 Visual Studio developer shell and the `windows-clang-coverage` preset. Generate the report from the same checkout as the coverage export. The generator looks for `coverage.info` beside `coverage.json`. Use `--lcov` to specify another file or generate from LCOV alone, using LLVM block and branch IDs in the details.

Open `report/index.html` in a browser. Use `--output` to change the destination, `--root` to report on another checkout, `--project-name` to override the checkout folder name, and `--badges-dir` to export the badge values. The report works offline. Repository document links use the GitHub origin and measured commit.

## Metrics and navigation

- **Code Coverage** divides covered lines and branch outcomes by all instrumented lines and branch outcomes. Folder totals add the counts from descendant files. If LLVM reports repeated template instantiations, an outcome counts as covered if any instance covered it. Yellow marks partial branch coverage. Select a fraction such as `1/4` to inspect its outcomes.
- **Requirement Coverage** counts intents linked to at least one requirement and requirements linked to at least one test, divided by the total number of intents and requirements. The card also gives both rates separately. Open a document to read its full Markdown. Linked specifications can switch between rendered Markdown and raw text. Linked tests show any `@verifies` rationale and expand to reveal the test source.
- **Source lines of code** sums Lizard's non-comment lines in `src/` and `include/`. See the [code metrics guide](../code_metrics/README.md) for the percentile calculation. The **P95 cognitive complexity** card also shows the highest function score.

**Dashboard** charts the four metrics in the browser's local time zone. Links can select a tab with `#coverage`, `#traceability`, or `#complexity`. The same keys work with `?tab=`.

## Publishing

Set **Settings → Pages → Source** to **GitHub Actions**. The Pages workflow uses the latest successful qualifying **Build & Tests** run. It publishes the report and branch snapshots containing production source, linked test source, document previews, and metrics. An active branch needs a successful same-repository push or manual run with the report workflow. Rerun the workflow to replace an expired snapshot. Deleted branches disappear after the next deployment.

The branch picker switches the three explorers. Dashboard and README badges use `main`. The workflow updates `code-metrics-history` with one measurement per commit and the current badge values. Reruns keep the original measurement time. Commits go through the workflow's `GITHUB_TOKEN` as `github-actions[bot]`. If you protect the history branch, allow the workflow's direct pushes.

Run the report tests with `python -m unittest discover -s tools/report`.
