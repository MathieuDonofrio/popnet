#!/usr/bin/env python3
"""Measure C/C++ source lines or percentile Cognitive Complexity."""

import argparse
import json
import math
from colorsys import hsv_to_rgb
from contextlib import redirect_stderr
from fractions import Fraction
from io import StringIO
from pathlib import Path


SCHEMA_VERSION = 1
EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}


def percentile_type(value: str) -> float:
    try:
        percentile = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("percentile must be a number") from exc
    if not 0 < percentile <= 100:
        raise argparse.ArgumentTypeError("percentile must be greater than 0 and at most 100")
    return percentile


def source_files(directories: list[Path]) -> list[Path]:
    candidates = {
        path for directory in set(directories)
        for path in directory.rglob("*")
        if path.suffix.lower() in EXTENSIONS
    }
    return sorted({path.resolve(strict=True) for path in candidates if path.is_file()})


def count_sloc(files: list[Path]) -> int:
    sloc, _ = analyze_sources(files)
    return sloc


def analyze_functions(files: list[Path]) -> list[dict]:
    _, functions = analyze_sources(files)
    return functions


def analyze_sources(files: list[Path]) -> tuple[int, list[dict]]:
    try:
        import lizard

        analyzer = lizard.FileAnalyzer(lizard.get_extensions(["cognitive"]))
    except ImportError as exc:
        raise RuntimeError("Install the required Lizard version with python -m pip install -r tools/code_metrics/requirements.txt") from exc

    sloc = 0
    functions = []
    for path in files:
        # Select Lizard's C/C++ reader for every accepted extension and casing.
        try:
            source = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{path} cannot be read as UTF-8: {exc}") from exc
        # Lizard can report a parser failure on stderr and return partial results.
        with redirect_stderr(StringIO()) as stderr:
            analysis = analyzer.analyze_source_code(str(path.with_suffix(".cpp")), source)
        if diagnostics := stderr.getvalue():
            raise RuntimeError(f"Lizard failed to analyze {path}: {diagnostics.strip()}")
        sloc += analysis.nloc
        functions.extend({
            "name": function.name,
            "file": path,
            "start_line": function.start_line,
            "end_line": function.end_line,
            "cognitive_complexity": function.cognitive_complexity,
        } for function in analysis.function_list)
    return sloc, functions


def cognitive_percentile(functions: list[dict], percentile: float) -> int:
    complexities = sorted(function["cognitive_complexity"] for function in functions)
    if not complexities:
        raise RuntimeError("No C/C++ functions found")
    # Avoid rounding a whole-number rank up due to binary floating-point arithmetic.
    rank = math.ceil(Fraction(str(percentile)) * len(complexities) / 100)
    return complexities[rank - 1]


def count_cognitive_complexity(files: list[Path], percentile: float) -> tuple[int, int]:
    functions = analyze_functions(files)
    return cognitive_percentile(functions, percentile), len(functions)


def function_details(functions: list[dict], root: Path) -> list[dict]:
    details = []
    for function in functions:
        path = function["file"]
        if path.is_relative_to(root):
            path = path.relative_to(root)
        details.append({**function, "file": path.as_posix()})
    return details


def make_badge(metric: str, value: int, percentile: float) -> dict:
    if metric == "sloc":
        label, message, color = "Source Lines of Code", f"{value:,}", "blue"
    else:
        label, message = f"P{str(percentile).removesuffix('.0')} Cognitive Complexity", str(value)
        # Sweep the hue from green through yellow to red over scores 0–30.
        progress = min(value / 30, 1)
        rgb = hsv_to_rgb((1 - progress) / 3, 0.8, 0.75)
        color = "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)
    return {"schemaVersion": 1, "label": label, "message": message, "color": color}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", choices=("sloc", "cognitive"), help="Metric to calculate")
    parser.add_argument("directories", nargs="+", type=Path, help="Directories to scan recursively")
    parser.add_argument("--percentile", type=percentile_type, metavar="P",
                        help="Cognitive Complexity percentile (default: 95)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    output.add_argument("--badge", action="store_true", help="Print Shields.io endpoint JSON")
    parser.add_argument("--functions", action="store_true",
                        help="Include individual functions with cognitive --json")
    args = parser.parse_intermixed_args()

    if args.functions and (args.metric != "cognitive" or not args.json):
        parser.error("--functions requires cognitive --json")
    if args.metric == "sloc" and args.percentile is not None:
        parser.error("--percentile is only valid for cognitive")
    percentile = args.percentile if args.percentile is not None else 95.0

    try:
        directories = [path.resolve() for path in args.directories]
        for directory in directories:
            if not directory.is_dir():
                parser.error(f"Not a directory: {directory}")
        files = source_files(directories)
        if not files:
            parser.error("No C/C++ source files found")

        report = {
            "schema_version": SCHEMA_VERSION,
            "metric": args.metric,
            "directories": [str(path) for path in directories],
            "files": len(files),
        }
        if args.metric == "sloc":
            value = count_sloc(files)
        elif args.functions:
            functions = analyze_functions(files)
            value = cognitive_percentile(functions, percentile)
            report.update(percentile=percentile, functions=len(functions),
                          function_details=function_details(functions, Path.cwd().resolve()))
        else:
            value, function_count = count_cognitive_complexity(files, percentile)
            report.update(percentile=percentile, functions=function_count)
        report["value"] = value
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if args.badge:
        report = make_badge(args.metric, value, percentile)
    elif not args.json:
        print(value)
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
