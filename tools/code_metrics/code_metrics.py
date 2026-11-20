#!/usr/bin/env python3
"""Measure C/C++ source lines or percentile cyclomatic complexity."""

import argparse
import json
import math
import subprocess
import tempfile
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
    # cloc omits zero-byte files, which contribute zero lines.
    nonempty_count = sum(path.stat().st_size > 0 for path in files)
    paths = [path.as_posix() for path in files]
    if any("\n" in path or "\r" in path for path in paths):
        raise ValueError("cloc cannot read filenames containing line breaks")

    with tempfile.TemporaryDirectory() as tmp:
        file_list = Path(tmp) / "files.txt"
        file_list.write_text("\n".join(paths) + "\n", encoding="utf-8")
        config = Path(tmp) / "options.txt"
        config.touch()  # Keep the user's cloc defaults out of the measurement.

        try:
            result = subprocess.run(
                [
                    "cloc", "--json", "--quiet", "--skip-uniqueness",
                    "--force-lang=C++", "--strip-str-comments",
                    f"--config={config}", f"--list-file={file_list}",
                ],
                capture_output=True, encoding="utf-8", errors="replace", check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("cloc is not installed or not on PATH") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or f"exit code {exc.returncode}"
            raise RuntimeError(f"cloc failed: {detail}") from exc

    diagnostics = result.stderr.strip()
    try:
        report = json.loads(result.stdout)
        if report == {} and nonempty_count == 0:
            return 0
        total = report["SUM"]
        counted, code = total["nFiles"], total["code"]
        if type(counted) is not int or type(code) is not int or code < 0:
            raise ValueError("invalid totals")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid cloc report: {diagnostics or exc}") from exc
    if counted != nonempty_count:
        detail = f": {diagnostics}" if diagnostics else ""
        raise RuntimeError(f"cloc counted {counted} of {nonempty_count} nonempty files{detail}")
    return code


def count_cc(files: list[Path], percentile: float) -> tuple[int, int]:
    try:
        import lizard
    except ImportError as exc:
        raise RuntimeError("Install Lizard with python -m pip install -r tools/code_metrics/requirements.txt") from exc

    complexities = []
    for path in files:
        # Select Lizard's C/C++ reader for every accepted extension and casing.
        try:
            source = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{path} cannot be read as UTF-8: {exc}") from exc
        # Lizard can report a parser failure on stderr and return partial results.
        with redirect_stderr(StringIO()) as stderr:
            analysis = lizard.analyze_file.analyze_source_code(str(path.with_suffix(".cpp")), source)
        if diagnostics := stderr.getvalue():
            raise RuntimeError(f"Lizard failed to analyze {path}: {diagnostics.strip()}")
        complexities.extend(function.cyclomatic_complexity for function in analysis.function_list)

    if not complexities:
        raise RuntimeError("No C/C++ functions found")
    complexities.sort()
    # Avoid rounding a whole-number rank up due to binary floating-point arithmetic.
    rank = math.ceil(Fraction(str(percentile)) * len(complexities) / 100)
    return complexities[rank - 1], len(complexities)


def make_badge(metric: str, value: int, percentile: float) -> dict:
    if metric == "sloc":
        label, message = "SLOC", f"{value:,}"
    else:
        label, message = f"P{str(percentile).removesuffix('.0')} CC", str(value)
    return {"schemaVersion": 1, "label": label, "message": message, "color": "blue"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", choices=("sloc", "cc"), help="Metric to calculate")
    parser.add_argument("directories", nargs="+", type=Path, help="Directories to scan recursively")
    parser.add_argument("--percentile", type=percentile_type, metavar="P", help="CC percentile (default: 95)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    output.add_argument("--badge", action="store_true", help="Print Shields.io endpoint JSON")
    args = parser.parse_intermixed_args()

    if args.metric == "sloc" and args.percentile is not None:
        parser.error("--percentile is only valid for cc")
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
        else:
            value, function_count = count_cc(files, percentile)
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
