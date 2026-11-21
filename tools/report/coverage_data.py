"""Read LLVM coverage exports for the dashboard's source explorer."""

import json
from collections import defaultdict
from pathlib import Path


def source_name(value, root, production):
    path = (root / value.replace("\\", "/")).resolve()
    if path.is_relative_to(root):
        name = path.relative_to(root).as_posix()
        if name in production:
            return name
    return None


def total(records):
    outcomes = [outcome for record in records for outcome in record["outcomes"].values()]
    return {"covered": sum(outcomes), "total": len(outcomes)}


def location(line, column, end_line, end_column, source):
    if not all(type(value) is int and value > 0 for value in (line, column, end_line, end_column)):
        raise ValueError("invalid source location")
    if line > end_line or end_line > len(source) or (line == end_line and column > end_column):
        raise ValueError("coverage location outside the source; rebuild coverage for this checkout")
    lines = source[line - 1:end_line]
    # LLVM reports byte offsets, including in UTF-8 source.
    if len(lines) == 1:
        expression = lines[0].encode("utf-8")[column - 1:end_column - 1].decode("utf-8", "replace")
    else:
        first = lines[0].encode("utf-8")[column - 1:].decode("utf-8", "replace")
        last = lines[-1].encode("utf-8")[:end_column - 1].decode("utf-8", "replace")
        expression = " ".join(part.strip() for part in [first, *lines[1:-1], last])
    return f"Column {column}" + (f" · {expression}" if expression else "")


def branch_outcomes(record):
    true, false = record[4:6]
    if any(type(count) is not int or count < 0 for count in (true, false)):
        raise ValueError("invalid branch counters")
    return {"True": true > 0, "False": false > 0}


def merge_record(lines, line, key, label, outcomes):
    # A source expression can appear in multiple template instantiations.
    record = lines[line].setdefault(key, {"label": label, "outcomes": {}})
    for name, covered in outcomes.items():
        record["outcomes"][name] = covered or record["outcomes"].get(name, False)


def read_json(path: Path, root: Path, production: dict) -> dict:
    coverage = {}
    name = None
    found = False
    try:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        if report["type"] != "llvm.coverage.json.export":
            raise ValueError("expected an llvm-cov JSON export")
        for export in report["data"]:
            for file in export["files"]:
                name = source_name(file["filename"], root, production)
                if name is None:
                    continue
                found = True
                source = production[name]
                branch_lines = coverage.setdefault(name, defaultdict(dict))
                for branch in file["branches"]:
                    label = location(*branch[:4], source)
                    merge_record(branch_lines, branch[0], tuple(branch[:4]), label, branch_outcomes(branch))
                for expansion in file.get("expansions", []):
                    site_range = expansion["source_region"][:4]
                    site = location(*site_range, source)
                    for branch in expansion["branches"]:
                        filename = expansion["filenames"][branch[6]].replace("\\", "/")
                        label = f"{site} · macro branch at {Path(filename).name}:{branch[0]}:{branch[1]}"
                        key = (*site_range, filename, *branch[:4])
                        merge_record(branch_lines, site_range[0], key, label, branch_outcomes(branch))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid LLVM coverage JSON{f' for {name}' if name else ''}: {exc}") from exc
    if not found:
        raise ValueError(f"{path}: no production coverage records match {root}")
    return {
        name: {line: [records[key] for key in sorted(records)] for line, records in lines.items()}
        for name, lines in coverage.items()
    }


def read_lcov(path: Path, root: Path, production: dict) -> dict:
    files = {}
    line_files = {}
    source = None
    branches = {}
    lines = {}
    ignored = {"TN", "LF", "LH", "FN", "FNDA", "FNF", "FNH", "BRF", "BRH"}
    for number, record in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not record.strip():
            continue
        field, _, value = record.partition(":")
        try:
            if field == "SF" and value and source is None:
                source = value
            elif field == "DA" and source is not None:
                row, count, *_ = value.split(",", 2)
                row, count = int(row), int(count)
                if row < 1 or count < 0 or row in lines:
                    raise ValueError("invalid or duplicate line outcome")
                lines[row] = count > 0
            elif field == "BRDA" and source is not None:
                line, block, branch, taken = value.split(",")
                key = (int(line), block, branch)
                count = 0 if taken == "-" else int(taken)
                if key[0] < 1 or not block or not branch or count < 0 or key in branches:
                    raise ValueError("invalid or duplicate branch outcome")
                branches[key] = count > 0
            elif record == "end_of_record" and source is not None:
                name = source_name(source, root, production)
                if name is not None:
                    outcomes = files.setdefault(name, {})
                    line_outcomes = line_files.setdefault(name, {})
                    for row, covered in lines.items():
                        if row > len(production[name]):
                            raise ValueError(f"coverage line outside {name}; rebuild coverage for this checkout")
                        line_outcomes[row] = covered or line_outcomes.get(row, False)
                    for key, covered in branches.items():
                        if key[0] > len(production[name]):
                            raise ValueError(f"coverage line outside {name}; rebuild coverage for this checkout")
                        outcomes[key] = covered or outcomes.get(key, False)
                source, branches, lines = None, {}, {}
            elif field not in ignored or (field != "TN" and source is None):
                raise ValueError(f"unexpected LCOV record {record!r}")
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if source is not None:
        raise ValueError(f"{path}: missing end_of_record for {source}")
    if not files:
        raise ValueError(f"{path}: no production coverage records match {root}")
    coverage = {}
    for name, outcomes in files.items():
        lines = defaultdict(list)
        for (line, block, branch), covered in sorted(outcomes.items()):
            lines[line].append({"label": f"Block {block}", "outcomes": {f"Branch {branch}": covered}})
        coverage[name] = lines
    return {
        "branches": coverage,
        "lines": {name: {row: [{"label": "", "outcomes": {"Executed": covered}}]
                         for row, covered in outcomes.items()} for name, outcomes in line_files.items()},
    }
