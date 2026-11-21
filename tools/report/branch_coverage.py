"""Exchange coverage-only snapshots between branch CI and the Pages publisher."""

from pathlib import PurePosixPath

from . import coverage_data


SCHEMA_VERSION = 1
PRODUCTION_ROOTS = ("src", "include")


def snapshot(report: dict, commit: str) -> dict:
    sources = {}
    for path in report["production"]:
        source = report["sources"][path]
        sources[path] = {
            "lines": source["lines"],
            "coverage": {metric: source.get("coverage", {}).get(metric, {}).get("lines", {})
                         for metric in coverage_data.METRICS},
        }
    return {"schema_version": SCHEMA_VERSION, "commit": commit, "sources": sources}


def validate_lines(records_by_line: dict, line_count: int, metric: str) -> None:
    for line, records in records_by_line.items():
        if not line.isascii() or not line.isdecimal() or line.startswith("0") or int(line) > line_count:
            raise ValueError("invalid coverage line number")
        if not isinstance(records, list) or not records:
            raise ValueError("invalid line coverage records")
        for record in records:
            if set(record) != {"label", "outcomes"} or not isinstance(record["label"], str):
                raise ValueError("invalid coverage record")
            outcomes = record["outcomes"]
            if not isinstance(outcomes, dict) or not outcomes or not all(type(value) is bool for value in outcomes.values()):
                raise ValueError("invalid coverage outcomes")
        if metric == "lines" and (len(records) != 1 or set(records[0]["outcomes"]) != {"Executed"}):
            raise ValueError("invalid line execution record")


def report_view(data: dict, commit: str) -> dict:
    """Validate branch data and rebuild the file and report totals."""
    if data["schema_version"] != SCHEMA_VERSION or data["commit"] != commit:
        raise ValueError("branch snapshot schema or commit does not match")
    if not data["sources"]:
        raise ValueError("branch snapshot has no production sources")
    sources = {}
    for path, source in sorted(data["sources"].items()):
        parts = PurePosixPath(path).parts
        if (len(parts) < 2 or parts[0] not in PRODUCTION_ROOTS or "/".join(parts) != path
                or ".." in parts or any(char in path for char in "\\:\x00")):
            raise ValueError(f"invalid production path: {path!r}")
        lines = source["lines"]
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            raise ValueError("invalid source lines")
        coverage = {metric: source["coverage"][metric] for metric in coverage_data.METRICS}
        for metric, records in coverage.items():
            validate_lines(records, len(lines), metric)
        sources[path] = {"lines": lines, "coverage": coverage_data.file_coverage(coverage)}
    return {"production": list(sources), "sources": sources, "coverage": coverage_data.report_coverage(sources)}
