"""Exchange report data between branch CI and the trusted Pages publisher."""

from pathlib import PurePosixPath

from tools.code_metrics.code_metrics import cognitive_percentile
from . import branch_coverage
from .markdown_preview import MARKDOWN


def snapshot(report: dict, commit: str) -> dict:
    trace = report["trace"]
    production = set(report["production"])

    def plain_document(document):
        return {**document, "markdown": {"text": document["markdown"]["text"]}}

    return {
        "code": branch_coverage.snapshot(report, commit),
        "test_sources": {path: source["lines"] for path, source in report["sources"].items()
                         if path not in production},
        "trace": {
            "intents": [plain_document(doc) for doc in trace["intents"]],
            "requirements": {key: plain_document(doc) for key, doc in trace["requirements"].items()},
            "specifications": {path: {"text": doc["text"], "title": doc["title"]}
                               for path, doc in trace["specifications"].items()},
        },
        "functions": report["functions"],
        "sloc": report["sloc"],
    }


def string(value):
    if not isinstance(value, str):
        raise ValueError("expected text in branch report")
    return value


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer in branch report")
    return value


def repo_path(value, scope=(), allow_fragment=False):
    path = string(value)
    if allow_fragment:
        path = path.partition("#")[0]
    parts = PurePosixPath(path).parts
    if (len(parts) <= len(scope) or parts[:len(scope)] != scope or "/".join(parts) != path
            or ".." in parts or any(char in path for char in "#\\:\x00")):
        raise ValueError(f"invalid branch report path: {value!r}")
    return value


def document(raw, kind):
    text = string(raw["markdown"]["text"])
    return {
        "id": string(raw["id"]),
        "title": string(raw["title"]),
        "path": repo_path(raw["path"], ("docs", kind + "s")),
        "type": kind,
        "markdown": {"text": text, "html": MARKDOWN.render(text)},
    }


def trace_view(raw, sources):
    intents = {}
    requirements = {}
    specifications = {}
    unassigned = []
    for entry in raw["intents"]:
        intent = document(entry, "intent")
        if intent["id"] in intents:
            raise ValueError("duplicate branch report intent")
        intents[intent["id"]] = {**intent, "requirements": []}

    for key, entry in sorted(raw["requirements"].items()):
        req = document(entry, "requirement")
        if req["id"] != key or key in intents:
            raise ValueError("invalid branch report requirement ID")
        req["intents"] = []
        for ref in entry["intents"]:
            intent = intents[string(ref["id"])]
            req["intents"].append({field: intent[field] for field in ("id", "title", "path", "type")})
            intent["requirements"].append(key)
        if not req["intents"]:
            unassigned.append(key)

        req["tests"] = []
        for ref in entry["tests"]:
            path = repo_path(ref["path"], ("tests",))
            line = integer(ref["line"], 1)
            if line > len(sources[path]["lines"]):
                raise ValueError("test line outside branch source")
            req["tests"].append({"name": string(ref["name"]), "path": path, "line": line})

        req["specifications"] = []
        for ref in entry["specifications"]:
            path = repo_path(ref["path"], allow_fragment=True)
            req["specifications"].append({"path": path})
            if path not in specifications:
                preview = raw["specifications"][path]
                text = string(preview["text"])
                specifications[path] = {"title": string(preview["title"]), "text": text}
                if PurePosixPath(path.partition("#")[0]).suffix.lower() == ".md":
                    specifications[path]["html"] = MARKDOWN.render(text)
        requirements[key] = req

    summary = {
        "intents_with_requirements": sum(bool(doc["requirements"]) for doc in intents.values()),
        "intents_total": len(intents),
        "requirements_with_tests": sum(bool(doc["tests"]) for doc in requirements.values()),
        "requirements_total": len(requirements),
    }
    return {
        "summary": summary,
        "combined": {"covered": summary["intents_with_requirements"] + summary["requirements_with_tests"],
                     "total": len(intents) + len(requirements)},
        "intents": list(intents.values()),
        "requirements": requirements,
        "specifications": specifications,
        "unassigned": unassigned,
    }


def report_view(data: dict, commit: str) -> dict:
    view = branch_coverage.report_view(data["code"], commit)
    sources = view["sources"]
    functions = []
    for raw in data["functions"]:
        path = string(raw["file"])
        start = integer(raw["start_line"], 1)
        end = integer(raw["end_line"], start)
        if end > len(sources[path]["lines"]):
            raise ValueError("function range outside branch source")
        functions.append({"name": string(raw["name"]), "file": path, "start_line": start,
                          "end_line": end, "cognitive_complexity": integer(raw["cognitive_complexity"])})
    if not functions:
        raise ValueError("branch report has no functions")

    tests = {}
    for path, lines in data["test_sources"].items():
        repo_path(path, ("tests",))
        if not isinstance(lines, list):
            raise ValueError("invalid branch test source")
        tests[path] = {"lines": [string(line) for line in lines]}
    trace = trace_view(data["trace"], tests)
    for req in trace["requirements"].values():
        for test in req["tests"]:
            sources[test["path"]] = tests[test["path"]]
    return {**view, "trace": trace, "functions": functions, "p95": cognitive_percentile(functions, 95),
            "sloc": integer(data["sloc"])}
