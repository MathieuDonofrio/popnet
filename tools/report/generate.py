#!/usr/bin/env python3
"""Generate a self-contained Popnet code report from LLVM coverage and the checkout."""

import argparse
import json
import os
import subprocess
import sys
from colorsys import hsv_to_rgb
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("template.html")
PRODUCTION = ("src", "include")
DATA_MARKER = '<script id="report-data" type="application/json"></script>'

sys.path.insert(0, str(ROOT))
from tools.code_metrics import code_metrics
from tools.report import branch_report, coverage_data, history
from tools.report.markdown_preview import MARKDOWN
from tools.trace.trace import Index, markdown_headings


def document_preview(text: str, fragment="") -> dict:
    lines = text.splitlines()
    headings = list(markdown_headings(lines))
    if not fragment:
        title = headings[0][2] if headings else ""
        return {"title": title, "text": text.strip()}
    for index, (number, depth, title, anchor) in enumerate(headings):
        if anchor == fragment:
            end = next((row - 1 for row, level, _, _ in headings[index + 1:] if level <= depth), len(lines))
            return {"title": title, "text": "\n".join(lines[number:end]).strip()}
    return {"title": "", "text": ""}


def specification_preview(root: Path, reference: str) -> dict:
    path, _, fragment = reference.partition("#")
    source = root / path
    preview = document_preview(source.read_text(encoding="utf-8-sig"), fragment)
    if source.suffix.lower() == ".md":
        preview["html"] = MARKDOWN.render(preview["text"])
    return preview


def artifact_markdown(doc) -> dict:
    text = doc.body
    return {"text": text, "html": MARKDOWN.render(text)}


def collect_traceability(root: Path) -> dict:
    index = Index(root)
    for diagnostic in index.diagnostics:
        if diagnostic["severity"] == "error":
            location = diagnostic["path"]
            if diagnostic["line"] is not None:
                location += f':{diagnostic["line"]}'
            raise ValueError(f'{location}: {diagnostic["message"]}')

    coverage = index.coverage()
    summary = coverage["summary"]
    combined = {
        "covered": summary["intents_with_requirements"] + summary["requirements_with_tests"],
        "total": summary["intents_total"] + summary["requirements_total"],
    }
    requirements = {
        req["id"]: index.show(req["id"])
        for req in coverage["requirements_without_intents"]
    }
    for intent in coverage["intents"]:
        intent["markdown"] = artifact_markdown(index.by_id[intent["id"]])
        for req in intent["requirements"]:
            if req["id"] not in requirements:
                requirements[req["id"]] = index.show(req["id"])
        intent["requirements"] = [req["id"] for req in intent["requirements"]]
    specifications = {}
    for req in requirements.values():
        req["markdown"] = artifact_markdown(index.by_id[req["id"]])
        for spec in req["specifications"]:
            reference = spec["path"]
            if reference not in specifications:
                specifications[reference] = specification_preview(root, reference)
    return {
        "summary": coverage["summary"],
        "combined": combined,
        "intents": coverage["intents"],
        "requirements": requirements,
        "specifications": specifications,
        "unassigned": [req["id"] for req in coverage["requirements_without_intents"]],
    }


def read_source(root: Path, name: str, scope: tuple[str, ...]) -> list[str]:
    path = (root / name).resolve()
    if not any(path.is_relative_to(root / directory) for directory in scope):
        raise ValueError(f"source is outside the report scope: {name}")
    text = path.read_text(encoding="utf-8-sig")
    return text.removesuffix("\n").split("\n") if text else []


def collect_report(root: Path, lcov_path: Path, json_path: Path | None = None) -> dict:
    root = root.resolve()
    files = code_metrics.source_files([root / directory for directory in PRODUCTION])
    production = {}
    for path in files:
        name = path.relative_to(root).as_posix()
        production[name] = read_source(root, name, PRODUCTION)

    coverage = coverage_data.read_lcov(lcov_path, root, production)
    if json_path is not None:
        detailed = coverage_data.read_json(json_path, root, production)
        if any(detailed.values()):
            coverage["branches"] = detailed
    sources = {name: {"lines": lines} for name, lines in production.items()}
    summary = {}
    for metric, file_records in coverage.items():
        for name, file_lines in file_records.items():
            if file_lines:
                sources[name].setdefault("coverage", {})[metric] = {
                    "lines": file_lines,
                    **coverage_data.total(record for records in file_lines.values() for record in records),
                }
        summary[metric] = coverage_data.total(
            record for file_lines in file_records.values() for records in file_lines.values() for record in records
        )
    summary["combined"] = coverage_totals(summary)
    for source in sources.values():
        if "coverage" in source:
            source["coverage"]["combined"] = coverage_totals(source["coverage"])

    trace = collect_traceability(root)
    for req in trace["requirements"].values():
        for test in req["tests"]:
            name = test["path"]
            if name not in sources:
                sources[name] = {"lines": read_source(root, name, ("tests",))}

    sloc, functions = code_metrics.analyze_sources(files)
    return {
        "sources": sources,
        "production": list(production),
        "coverage": summary,
        "trace": trace,
        "sloc": sloc,
        "p95": code_metrics.cognitive_percentile(functions, 95),
        "functions": code_metrics.function_details(functions, root),
    }


def coverage_totals(metrics: dict) -> dict:
    return {key: sum(metric[key] for metric in metrics.values()) for key in ("covered", "total")}


def metric_badges(report: dict) -> dict:
    def percentage_badge(label, counts):
        total = counts["total"]
        fraction = counts["covered"] / total if total else None
        message, color = "—", "lightgrey"
        if fraction is not None:
            message = f"{100 * fraction:.1f}%"
            rgb = hsv_to_rgb(fraction / 3, 0.8, 0.75)
            color = "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)
        return {"schemaVersion": 1, "label": label, "message": message, "color": color}

    return {
        "coverage": percentage_badge("Code Coverage", report["coverage"]["combined"]),
        "traceability": percentage_badge("Requirement Coverage", report["trace"]["combined"]),
        "sloc": code_metrics.make_badge("sloc", report["sloc"], 95),
        "cognitive-complexity": code_metrics.make_badge("cognitive", report["p95"], 95),
    }


def render_report(data: dict) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(DATA_MARKER) != 1:
        raise ValueError("expected one report data element in the dashboard template")
    # Escape '<' so even a literal </script> in source or metadata stays inside the JSON.
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace(DATA_MARKER, f'{DATA_MARKER.removesuffix("</script>")}{payload}</script>')


def repository_browse_base(remote: str, actions_server: str = "") -> str | None:
    # Git can store an SSH remote in scp syntax, without a URL scheme.
    if remote.startswith("git@") and ":" in remote:
        host, path = remote.removeprefix("git@").split(":", 1)
        remote = urlunsplit(("ssh", host, "/" + path, "", ""))

    try:
        source = urlsplit(remote)
        server = urlsplit(actions_server)
        host = source.hostname
        if source.scheme not in {"http", "https", "ssh"} or not host:
            return None
        if host == server.hostname and server.scheme in {"http", "https"}:
            scheme, address = server.scheme, server.netloc.rsplit("@", 1)[-1]
        elif host == "github.com":
            scheme, address = "https", host
        else:
            return None
    except ValueError:
        return None

    parts = source.path.rstrip("/").removesuffix(".git").strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return urlunsplit((scheme, address, "/" + "/".join(parts) + "/blob/", "", ""))


def git_metadata() -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                                capture_output=True, encoding="utf-8")
        return result.stdout.strip()

    try:
        origin = git("remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        origin = ""
    return {
        "commit": git("rev-parse", "HEAD"),
        "ref": git("rev-parse", "--abbrev-ref", "HEAD"),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document_base": repository_browse_base(origin, os.environ.get("GITHUB_SERVER_URL", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-json", type=Path, help="Path to LLVM coverage.json for branch details")
    parser.add_argument("--lcov", type=Path, help="Path to LLVM coverage.info (defaults to a sibling of the JSON file)")
    parser.add_argument("--output", type=Path, default=Path("report/index.html"))
    parser.add_argument("--badges-dir", type=Path, help="Write Shields.io badge JSON alongside the report")
    parser.add_argument("--metrics-snapshot", type=Path, help="Write this run's metrics for the history job")
    parser.add_argument("--branch-snapshot", type=Path, help="Write report data for branch browsing")
    args = parser.parse_args()
    try:
        if not (args.lcov or args.coverage_json):
            parser.error("provide --lcov or --coverage-json")
        lcov = args.lcov or args.coverage_json.with_name("coverage.info")
        data = collect_report(ROOT, lcov, args.coverage_json)
        badges = metric_badges(data)
        revision = git_metadata()
        measurement = history.snapshot(data, revision)
        html = render_report({**data, "revision": revision, "badges": badges})
        html = history.embed_history(html, history.record(history.empty_history(), measurement))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")
        if args.badges_dir is not None:
            args.badges_dir.mkdir(parents=True, exist_ok=True)
            for name, badge in badges.items():
                (args.badges_dir / f"{name}.json").write_text(json.dumps(badge, indent=2) + "\n", encoding="utf-8")
        if args.metrics_snapshot is not None:
            args.metrics_snapshot.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_snapshot.write_text(json.dumps(measurement, indent=2) + "\n", encoding="utf-8")
        if args.branch_snapshot is not None:
            args.branch_snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot = branch_report.snapshot(data, revision["commit"])
            args.branch_snapshot.write_text(json.dumps(snapshot, separators=(",", ":")) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
