"""Record dashboard metrics and embed the history in the static report."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
HISTORY_ELEMENT = re.compile(r'(<script id="report-history" type="application/json">).*?(</script>)', re.DOTALL)


def snapshot(report: dict, revision: dict) -> dict:
    return {
        "commit": revision["commit"],
        "recorded_at": revision["generated"],
        "code_coverage": report["coverage"]["combined"],
        "requirement_coverage": report["trace"]["combined"],
        "sloc": report["sloc"],
        "p95": report["p95"],
    }


def measurement_time(entry: dict) -> datetime:
    timestamp = datetime.fromisoformat(entry["recorded_at"])
    if timestamp.tzinfo is None:
        raise ValueError("measurement timestamp needs a time zone")
    return timestamp.astimezone(timezone.utc)


def record(history: dict, entry: dict) -> dict:
    if history.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported metrics history schema")

    by_commit = {}
    for item in history["measurements"]:
        if item["commit"] in by_commit:
            raise ValueError(f'duplicate measurement for {item["commit"]}')
        by_commit[item["commit"]] = item

    commit = entry["commit"]
    previous = by_commit.get(commit)
    # Reruns refresh the values while retaining the first measurement's date.
    by_commit[commit] = {**entry, "recorded_at": previous["recorded_at"]} if previous else entry
    return {
        "schema_version": SCHEMA_VERSION,
        "measurements": sorted(by_commit.values(), key=lambda item: (measurement_time(item), item["commit"])),
    }


def empty_history() -> dict:
    return {"schema_version": SCHEMA_VERSION, "measurements": []}


def embed_history(html: str, history: dict) -> str:
    payload = json.dumps(history["measurements"], separators=(",", ":")).replace("<", "\\u003c")
    result, replaced = HISTORY_ELEMENT.subn(lambda match: match[1] + payload + match[2], html)
    if replaced != 1:
        raise ValueError("expected one report history element")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    try:
        previous = json.loads(args.history.read_text(encoding="utf-8")) if args.history.exists() else empty_history()
        current = json.loads(args.snapshot.read_text(encoding="utf-8"))
        history = record(previous, current)
        report = embed_history(args.report.read_text(encoding="utf-8"), history)
        args.history.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        args.report.write_text(report, encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
