# Interactive Code Report

Generate LLVM coverage for the current checkout using the repository's coverage preset. With Python 3.10 or newer, run the following commands from the repository root:

```sh
cmake --preset linux-clang-coverage
cmake --build build/linux-clang-coverage --target coverage
python -m pip install -r tools/trace/requirements.txt -r tools/code_metrics/requirements.txt -r tools/report/requirements.txt
python tools/report/generate.py --coverage-json build/linux-clang-coverage/coverage/coverage.json
```

On Windows, use an x64 Visual Studio developer shell and replace `linux-clang-coverage` with `windows-clang-coverage`. Generate coverage and the dashboard on the same checkout so source paths and line numbers agree. With `--coverage-json`, the generator reads the adjacent `coverage.info` for line data. Use `--lcov` to override its location. An LCOV export alone also produces combined coverage, with LLVM's numeric block and branch IDs in branch details.

Open `report/index.html` locally. Use `--output` to write a different file. The Build & Tests workflow also publishes the page as the `popnet-code-report` artifact. Download and extract that artifact from the workflow run to open it locally. The page embeds all source and metric data it needs. Repository document links use the checkout's GitHub `origin` and commit and require a connection. If the report cannot determine a GitHub repository, it shows the document path.

- **Combined coverage** adds covered lines and covered branch outcomes and divides that sum by all instrumented lines and branch outcomes. The report, folder, and file percentages use that definition. The summary card also shows separate line and branch fractions and percentages. Line number colors show whether each instrumented line ran, and yellow rows indicate partial branch coverage. A branch fraction like `1/4` means one of four recorded outcomes on that line executed. Select it to inspect the branch outcomes directly below that line.
- **Combined traceability coverage** adds intents with at least one linked requirement and requirements with at least one linked test, then divides by the total number of intents and requirements. Each document counts once in its group. Traceability uses relationships declared in repository documents and `@verifies` annotations. The details pane renders the full Markdown body of the selected intent or requirement. Expand a Markdown specification to switch between rendered and raw text. Markdown tables and TeX math delimited by `$...$` or `$$...$$` render offline. Math uses the browser's MathML support, and formulas the converter cannot render appear as text. Embedded images are disabled. Expand a linked test to see its source in place, scrolled to the test declaration. **Show gaps only** displays missing intent-to-requirement and requirement-to-test links.
- **Source lines of code** sums Lizard's per-file NLOC on the production C/C++ files under `src/` and `include/`.
- **P95 cognitive complexity** uses the nearest-rank percentile of the same Lizard scores shown in the function table. The card also shows the highest function score.

Folder coverage combines the covered/total counts for descendant files. For repeated template instantiations, an outcome counts as covered at a source location if LLVM reports it covered in any instance. Dashboard percentages can differ from other LLVM summaries.

The traceability card shows the fraction and percentage of requirements with at least one linked test and of intents with at least one linked requirement.

Open the dashboard with `#coverage`, `#traceability`, or `#complexity` appended to the page URL to open that tab. Use `--badges-dir` to export the four README badge values with the dashboard. The Build & Tests workflow publishes them from the same report run on `main`.

The workflow deploys the report to GitHub Pages on `main`. Set the repository's **Settings → Pages → Source** to **GitHub Actions** before the first deployment.

The report embeds production C/C++ sources, test files referenced by traceability, and the linked documentation previews. It represents one checkout and coverage run.

Run the generator tests with the Python dependencies installed:

```sh
python -m unittest discover -s tools/report
```
