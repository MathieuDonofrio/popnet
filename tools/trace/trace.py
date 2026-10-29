#!/usr/bin/env python3
"""Navigate and validate Popnet's documented traceability relationships.

Requires PyYAML: python -m pip install PyYAML
Run from any directory: python <repository>/tools/trace/trace.py <command>
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: python -m pip install PyYAML")


SCHEMA_VERSION = 1
KINDS = {
    "intent": ("docs/intents", "INT", ("Problem", "Desired outcome", "Scope", "Success assessment")),
    "requirement": ("docs/requirements", "REQ", ("Obligation",)),
    "adr": ("docs/architecture", "ADR", ("Context", "Decision", "Rationale", "Consequences")),
}
ID_RE = re.compile(r"^(INT|REQ|ADR)-\d{4}$")
TEST_RE = re.compile(r'^\s*TEST_CASE\s*\(\s*"((?:\\.|[^"\\])*)"\s*\)', re.MULTILINE | re.DOTALL)
ANNOTATION_RE = re.compile(r'^\s*//\s*@verifies\b(.*)$')
VALID_ANNOTATION_RE = re.compile(r'^\s*(REQ-\d{4})(?:\s*:\s*.*)?\s*$')
HEADING_RE = re.compile(r'^##\s+(.+?)\s*#*\s*$')
FENCE_RE = re.compile(r'^\s*(`{3,}|~{3,})')
FILENAME_RE = re.compile(r'^(INT|REQ|ADR)-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$')


class StrictLoader(yaml.SafeLoader):
    """Safe YAML loading, rejecting duplicate mapping keys."""


def unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(None, None, "unhashable YAML key", key_node.start_mark) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate YAML key: {key}", key_node.start_mark)
        result[key] = loader.construct_object(value_node)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


@dataclass
class Document:
    kind: str
    path: str
    title: str
    text: str
    id: str | None = None
    metadata: dict = field(default_factory=dict)
    field_lines: dict = field(default_factory=dict)

    def brief(self):
        return {"id": self.id, "type": self.kind, "title": self.title, "path": self.path}


@dataclass
class TestCase:
    path: str
    line: int
    name: str
    requirements: list[str]

    def brief(self):
        return {"name": self.name, "path": self.path, "line": self.line}


class Index:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.documents = []
        self.by_id = {}
        self.tests = []
        self.tests_by_requirement = defaultdict(list)
        self.requirements_by_intent = defaultdict(list)
        self.diagnostics = []
        self._load()

    def diagnostic(self, code, message, path, line=None, severity="error"):
        self.diagnostics.append({
            "severity": severity, "code": code, "path": path,
            "line": line, "message": message,
        })

    def _load(self):
        if not (self.root / "docs/intents").is_dir() or not (self.root / "docs/requirements").is_dir():
            raise ValueError(f"not a Popnet repository: missing docs/intents or docs/requirements in {self.root}")
        for kind, (folder, _, _) in KINDS.items():
            base = self.root / folder
            if not base.exists():
                continue
            paths = sorted(base.rglob("ADR-*.md" if kind == "adr" else "*.md"))
            for path in paths:
                self._load_document(path, kind)
        specs = self.root / "docs/specifications"
        if specs.exists():
            for path in sorted(specs.rglob("*.md")):
                text = path.read_text(encoding="utf-8-sig")
                title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
                self.documents.append(Document("specification", self.rel(path), title, text))
        self._check_references()
        tests = self.root / "tests"
        if not tests.is_dir():
            self.diagnostic("MISSING_TEST_DIRECTORY", "tests/ directory does not exist", "tests")
        else:
            for path in sorted(t for t in tests.rglob("*") if t.suffix in (".cpp", ".cc", ".cxx")):
                self._load_tests(path)
        for test in self.tests:
            for req_id in test.requirements:
                self.tests_by_requirement[req_id].append(test)
        self.diagnostics.sort(key=lambda d: (d["path"], d["line"] or 0, d["code"]))

    def rel(self, path):
        return path.relative_to(self.root).as_posix()

    def _load_document(self, path, kind):
        location = self.rel(path)
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        metadata, fields, body_start = {}, {}, 0
        if not lines or lines[0].strip() != "---":
            self.diagnostic("MISSING_FRONT_MATTER", "missing opening YAML front matter delimiter", location, 1)
        else:
            end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
            if end is None:
                self.diagnostic("INVALID_FRONT_MATTER", "missing closing YAML front matter delimiter", location, 1)
            else:
                body_start = end + 1
                for lineno in range(1, end):
                    match = re.match(r'^([\w-]+)\s*:', lines[lineno])
                    if match:
                        fields[match.group(1)] = lineno + 1
                try:
                    parsed = yaml.load("\n".join(lines[1:end]), Loader=StrictLoader)
                    if not isinstance(parsed, dict):
                        raise ValueError("front matter must be a YAML mapping")
                    metadata = parsed
                except (yaml.YAMLError, ValueError) as exc:
                    self.diagnostic("INVALID_YAML", f"invalid front matter: {exc}", location, 2)
        prefix = KINDS[kind][1]
        declared_id = metadata.get("id")
        declared_title = metadata.get("title")
        for key in ("id", "title"):
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                self.diagnostic("MISSING_FIELD", f"'{key}' must be a nonempty string", location, fields.get(key, 1))
        if isinstance(declared_id, str) and not re.fullmatch(rf'{prefix}-\d{{4}}', declared_id):
            self.diagnostic("INVALID_ID", f"invalid {kind} ID: {declared_id}", location, fields.get("id", 1))
        match = re.match(rf'^({prefix}-\d{{4}})-', path.name)
        if not FILENAME_RE.fullmatch(path.name) or (match and declared_id != match.group(1)):
            self.diagnostic("INVALID_FILENAME", f"filename must be {prefix}-NNNN-short-title.md and match its declared ID", location, 1)
        for required in ("intents", "specifications") if kind == "requirement" else ():
            if not isinstance(metadata.get(required), list):
                self.diagnostic("INVALID_FIELD", f"'{required}' must be a YAML list", location, fields.get(required, 1))
        self._check_sections(lines, body_start, kind, location)
        doc = Document(kind, location, declared_title if isinstance(declared_title, str) else path.stem,
                       text, declared_id if isinstance(declared_id, str) else None, metadata, fields)
        self.documents.append(doc)
        if doc.id:
            if doc.id in self.by_id:
                self.diagnostic("DUPLICATE_ID", f"{doc.id} also declared in {self.by_id[doc.id].path}", location, fields.get("id", 1))
            else:
                self.by_id[doc.id] = doc

    def _check_sections(self, lines, start, kind, location):
        sections = {}
        current = None
        fence = None
        for i in range(start, len(lines)):
            line = lines[i]
            f = FENCE_RE.match(line)
            if f:
                marker = f.group(1)
                if fence is None:
                    fence = marker
                elif marker[0] == fence[0] and len(marker) >= len(fence):
                    fence = None
                continue
            if fence:
                continue
            heading = HEADING_RE.match(line)
            if heading:
                current = heading.group(1).strip()
                if current in sections:
                    self.diagnostic("DUPLICATE_SECTION", f"duplicate section: {current}", location, i + 1)
                else:
                    sections[current] = {"line": i + 1, "body": []}
            elif current:
                sections[current]["body"].append(line)
        for section in KINDS[kind][2]:
            if section not in sections:
                self.diagnostic("MISSING_SECTION", f"missing required section: {section}", location)
            elif not "".join(sections[section]["body"]).strip():
                self.diagnostic("EMPTY_SECTION", f"empty required section: {section}", location, sections[section]["line"])

    def _reference_line(self, doc, field, value, occurrence=1):
        lines = doc.text.splitlines()
        at = doc.field_lines.get(field, 1)
        for idx in range(at, len(lines)):
            if re.match(r'^\S[^:]*:', lines[idx]) and idx + 1 > at:
                break
            if lines[idx].strip().removeprefix("- ").strip().strip('"\'') == value:
                occurrence -= 1
                if occurrence == 0:
                    return idx + 1
        return at

    def _check_references(self):
        for doc in self.documents:
            if doc.kind != "requirement":
                continue
            for field_name in ("intents", "specifications"):
                values = doc.metadata.get(field_name)
                if not isinstance(values, list):
                    continue
                seen = set()
                occurrences = defaultdict(int)
                for value in values:
                    if isinstance(value, str):
                        occurrences[value] += 1
                    loc = (self._reference_line(doc, field_name, value, occurrences[value])
                           if isinstance(value, str) else doc.field_lines.get(field_name))
                    if not isinstance(value, str) or not value.strip():
                        self.diagnostic("INVALID_REFERENCE", f"{field_name} entries must be nonempty strings", doc.path, loc)
                        continue
                    if value in seen:
                        self.diagnostic("DUPLICATE_REFERENCE", f"duplicate {field_name} reference: {value}", doc.path, loc)
                        continue
                    seen.add(value)
                    if field_name == "intents":
                        if not re.fullmatch(r'INT-\d{4}', value) or value not in self.by_id or self.by_id[value].kind != "intent":
                            self.diagnostic("UNKNOWN_INTENT", f"unknown intent: {value}", doc.path, loc)
                        else:
                            self.requirements_by_intent[value].append(doc)
                    else:
                        file_part, sep, fragment = value.partition("#")
                        target = (self.root / file_part).resolve()
                        if not file_part or not target.is_relative_to(self.root) or not target.is_file():
                            self.diagnostic("UNKNOWN_SPECIFICATION", f"missing referenced file: {value}", doc.path, loc)
                        elif sep and (not fragment or not self._has_heading(target, fragment)):
                            self.diagnostic("UNKNOWN_SECTION", f"missing referenced section: {value}", doc.path, loc)

    @staticmethod
    def _has_heading(path, fragment):
        def slug(title):
            return re.sub(r'\s+', '-', re.sub(r'[^\w -]', '', title.casefold()).strip())
        headings = []
        fenced = None
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = FENCE_RE.match(line)
            if match:
                marker = match.group(1)
                if fenced is None:
                    fenced = marker
                elif marker[0] == fenced[0] and len(marker) >= len(fenced):
                    fenced = None
                continue
            if fenced:
                continue
            heading = re.match(r'^#{1,6}\s+(.*?)\s*#*\s*$', line)
            if heading:
                headings.append(slug(heading.group(1)))
        # Support GitHub's automatic suffixes for repeated headings.
        counts = defaultdict(int)
        for name in headings:
            suffix = counts[name]
            counts[name] += 1
            if (name if suffix == 0 else f"{name}-{suffix}") == fragment:
                return True
        return False

    def _load_tests(self, path):
        location = self.rel(path)
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        annotations = {}
        for idx, line in enumerate(lines):
            match = ANNOTATION_RE.match(line)
            if match:
                annotations[idx] = match.group(1)
        used = set()
        for match in TEST_RE.finditer(text):
            start = text.count("\n", 0, match.start())
            cursor = start - 1
            pending = []
            while cursor >= 0:
                if not lines[cursor].strip():
                    cursor -= 1
                elif cursor in annotations:
                    pending.append(cursor)
                    cursor -= 1
                else:
                    break
            pending.reverse()
            used.update(pending)
            references = []
            for idx in pending:
                a = VALID_ANNOTATION_RE.fullmatch(annotations[idx])
                if a is None:
                    self.diagnostic("INVALID_ANNOTATION", "expected // @verifies REQ-NNNN: optional rationale", location, idx + 1)
                    continue
                req = a.group(1)
                if req not in self.by_id or self.by_id[req].kind != "requirement":
                    self.diagnostic("UNKNOWN_REQUIREMENT", f"unknown requirement in @verifies: {req}", location, idx + 1)
                if req in references:
                    self.diagnostic("DUPLICATE_ANNOTATION", f"duplicate @verifies for {req} in test case", location, idx + 1)
                else:
                    references.append(req)
            if not pending:
                self.diagnostic("MISSING_ANNOTATION", "TEST_CASE must have at least one @verifies annotation", location, start + 1)
            self.tests.append(TestCase(location, start + 1, match.group(1), references))
        for idx in sorted(set(annotations) - used):
            self.diagnostic("MISPLACED_ANNOTATION", "@verifies annotation is not directly associated with a TEST_CASE", location, idx + 1)

    def _tests_for(self, req_id):
        return [test.brief() for test in self.tests_by_requirement[req_id]]

    def search(self, query):
        query = query.strip().casefold()
        if not query:
            return []
        words = query.split()
        matches = []
        for doc in self.documents:
            title = doc.title.casefold()
            artifact_id = (doc.id or "").casefold()
            path = doc.path.casefold()
            body = doc.text.casefold()
            if artifact_id == query:
                score = 0
            elif title == query:
                score = 1
            elif query in title:
                score = 2
            elif all(word in title for word in words):
                score = 3
            elif query in path:
                score = 4
            elif query in body or all(word in body for word in words):
                score = 5
            else:
                continue
            matches.append((score, doc.path, doc.brief()))
        return [entry for _, _, entry in sorted(matches)]

    def show(self, artifact_id):
        doc = self.by_id.get(artifact_id)
        if doc is None:
            return None
        result = doc.brief()
        if doc.kind == "intent":
            result["requirements"] = [
                {**req.brief(), "tests": self._tests_for(req.id)}
                for req in sorted(self.requirements_by_intent[artifact_id], key=lambda r: r.id)
            ]
        elif doc.kind == "requirement":
            result["intents"] = [
                self.by_id[value].brief() for value in doc.metadata.get("intents", [])
                if isinstance(value, str) and value in self.by_id and self.by_id[value].kind == "intent"
            ]
            result["specifications"] = [
                {"path": value} for value in doc.metadata.get("specifications", []) if isinstance(value, str)
            ]
            result["tests"] = self._tests_for(artifact_id)
        return result

    def coverage(self, intent_id=None):
        intents = sorted((d for d in self.documents if d.kind == "intent" and (intent_id is None or d.id == intent_id)), key=lambda d: d.id or "")
        requirements = sorted((d for d in self.documents if d.kind == "requirement" and d.id), key=lambda d: d.id)
        if intent_id is not None:
            req_ids = {r.id for r in self.requirements_by_intent[intent_id]}
            requirements = [r for r in requirements if r.id in req_ids]
        linked = sum(bool(self.tests_by_requirement[d.id]) for d in requirements)
        total = len(requirements)
        result = {
            "summary": {
                "intents_total": len(intents),
                "intents_with_requirements": sum(bool(self.requirements_by_intent[d.id]) for d in intents),
                "requirements_total": total,
                "requirements_with_tests": linked,
                "test_traceability_percent": round(100 * linked / total, 1) if total else 0.0,
            },
            "intents": [
                {**intent.brief(), "requirements": [
                    {**req.brief(), "tests": self._tests_for(req.id)}
                    for req in sorted(self.requirements_by_intent[intent.id], key=lambda r: r.id)
                ]} for intent in intents
            ],
            "requirements_without_tests": [d.brief() for d in requirements if not self.tests_by_requirement[d.id]],
        }
        if intent_id is None:
            result["requirements_without_intents"] = [
                d.brief() for d in requirements if not any(
                    d in members for members in self.requirements_by_intent.values()
                )
            ]
        return result


def print_show(info):
    print(f"{info['id']} — {info['title']}\n{info['path']}")
    if "intents" in info:
        print("\nIntents:")
        for intent in info["intents"]:
            print(f"  {intent['id']} — {intent['title']} ({intent['path']})")
        print("\nSpecifications:")
        for spec in info["specifications"]:
            print(f"  {spec['path']}")
        print("\nTests:")
        for test in info["tests"]:
            print(f"  {test['path']}:{test['line']} — {test['name']}")
    if "requirements" in info:
        print("\nRequirements:")
        for req in info["requirements"]:
            print(f"  {req['id']} — {req['title']} ({req['path']})")
            for test in req["tests"]:
                print(f"    {test['path']}:{test['line']} — {test['name']}")


def print_coverage(info):
    summary = info["summary"]
    print("Traceability coverage")
    print(f"Intents with requirements: {summary['intents_with_requirements']}/{summary['intents_total']}")
    print(f"Requirements with tests: {summary['requirements_with_tests']}/{summary['requirements_total']} "
          f"({summary['test_traceability_percent']:g}%)")
    for intent in info["intents"]:
        print(f"\n{intent['id']} — {intent['title']}")
        if not intent["requirements"]:
            print("  (no linked requirements)")
        for req in intent["requirements"]:
            n = len(req["tests"])
            print(f"  {req['id']} — {req['title']} ({n} test{'s' if n != 1 else ''})")
            for test in req["tests"]:
                print(f"    {test['path']}:{test['line']} — {test['name']}")
    if "requirements_without_intents" in info and info["requirements_without_intents"]:
        print("\nRequirements without intents:")
        for req in info["requirements_without_intents"]:
            print(f"  {req['id']} — {req['title']}")
    if info["requirements_without_tests"]:
        print("\nRequirements without tests:")
        for req in info["requirements_without_tests"]:
            print(f"  {req['id']} — {req['title']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "show", "coverage", "validate"):
        sub = commands.add_parser(name)
        if name == "search":
            sub.add_argument("query")
        elif name == "show":
            sub.add_argument("id")
        elif name == "coverage":
            sub.add_argument("intent_id", nargs="?")
            sub.add_argument("--badge", action="store_true", help="print Shields.io endpoint JSON")
        sub.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    if args.command == "coverage" and args.badge and args.json:
        parser.error("--badge and --json cannot be combined")
    root = Path(__file__).resolve().parents[2]
    try:
        index = Index(root)
    except (OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"schema_version": SCHEMA_VERSION,
                              "error": {"code": "LOAD_FAILED", "message": str(exc)}}, indent=2))
        else:
            print(f"trace: {exc}", file=sys.stderr)
        return 2
    status = 0
    if args.command == "search":
        results = index.search(args.query)
        result = {"results": results}
        if not args.json:
            for item in results:
                print(f"{item['id'] or '-':<9} {item['title']}\n          {item['path']}")
    elif args.command == "show":
        info = index.show(args.id)
        if info is None:
            if args.json:
                print(json.dumps({"schema_version": SCHEMA_VERSION, "error": {"code": "NOT_FOUND", "message": f"unknown artifact ID: {args.id}"}}, indent=2))
            else:
                print(f"unknown artifact ID: {args.id}", file=sys.stderr)
            return 1
        result = info
        if not args.json:
            print_show(info)
    elif args.command == "coverage":
        if args.intent_id is not None and (args.intent_id not in index.by_id or index.by_id[args.intent_id].kind != "intent"):
            if args.json:
                print(json.dumps({"schema_version": SCHEMA_VERSION, "error": {"code": "NOT_FOUND", "message": f"unknown intent: {args.intent_id}"}}, indent=2))
            else:
                print(f"unknown intent: {args.intent_id}", file=sys.stderr)
            return 1
        result = index.coverage(args.intent_id)
        if args.badge:
            percent = result["summary"]["test_traceability_percent"]
            color = "brightgreen" if percent >= 90 else "yellowgreen" if percent >= 75 else "yellow" if percent >= 50 else "red"
            print(json.dumps({"schemaVersion": 1, "label": "traceability", "message": f"{percent:g}%", "color": color}, indent=2))
            return 0
        if not args.json:
            print_coverage(result)
    else:
        errors = sum(d["severity"] == "error" for d in index.diagnostics)
        warnings = sum(d["severity"] == "warning" for d in index.diagnostics)
        result = {"errors": errors, "warnings": warnings, "diagnostics": index.diagnostics}
        if not args.json:
            for diagnostic in index.diagnostics:
                location = diagnostic["path"] + (f":{diagnostic['line']}" if diagnostic["line"] else "")
                print(f"{diagnostic['severity'].upper()} {location} [{diagnostic['code']}] {diagnostic['message']}")
            print(f"Validation: {errors} error(s), {warnings} warning(s)")
        status = 1 if errors else 0
    if args.json:
        print(json.dumps({"schema_version": SCHEMA_VERSION, **result}, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    sys.exit(main())
