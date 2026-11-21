"""Assemble the Pages site from the main report and branch report artifacts."""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from io import BytesIO
from itertools import count
from pathlib import Path
from urllib.parse import urlencode
from zipfile import BadZipFile, ZipFile

from . import branch_report


BRANCH_ELEMENT = re.compile(r'(<script id="report-branches" type="application/json">)\[\](</script>)')
BRANCH_ARTIFACT = "branch-report"
BRANCH_FILE = "branch-report.json"
REPORT_ARTIFACT = "popnet-code-report"
GENERATED_BRANCHES = {"badge", "badges", "code-metrics-history", "gh-pages"}
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024


class Actions:
    def __init__(self, repository: str):
        self.repository = repository
        self.artifact_index = {}

    def get(self, path: str, **query) -> bytes:
        endpoint = f"repos/{self.repository}/{path}"
        if query:
            endpoint += "?" + urlencode(query)
        return subprocess.check_output(["gh", "api", endpoint])

    def pages(self, path: str, key=None, **query):
        for page in count(1):
            response = json.loads(self.get(path, per_page=100, page=page, **query))
            items = response if key is None else response[key]
            yield from items
            if len(items) < 100:
                return

    def branches(self):
        return self.pages("branches")

    def artifacts(self, branch: str, name: str):
        if name not in self.artifact_index:
            index = defaultdict(list)
            for artifact in self.pages("actions/artifacts", "artifacts", name=name):
                run = artifact.get("workflow_run")
                if (artifact["name"] == name and not artifact["expired"] and run
                        and run["repository_id"] == run["head_repository_id"]):
                    index[run["head_branch"]].append(artifact)
            self.artifact_index[name] = index

        candidates = self.artifact_index[name].get(branch, [])
        for artifact in sorted(candidates, key=lambda item: (item["created_at"], item["id"]), reverse=True):
            run_id = artifact["workflow_run"]["id"]
            run = json.loads(self.get(f"actions/runs/{run_id}"))
            origin = run["head_repository"] or {}
            if (run["head_branch"] != branch or run["conclusion"] != "success"
                    or run["event"] not in ("push", "workflow_dispatch")
                    or run["path"].partition("@")[0] != ".github/workflows/ci.yml"
                    or origin.get("full_name", "").casefold() != self.repository.casefold()):
                continue
            yield run, artifact

    def read_artifact(self, artifact: dict, filename: str) -> bytes:
        if artifact["size_in_bytes"] > MAX_ARCHIVE_BYTES:
            raise ValueError("report artifact is too large")
        archive = self.get(f'actions/artifacts/{artifact["id"]}/zip')
        with ZipFile(BytesIO(archive)) as files:
            member = files.getinfo(filename)
            if member.file_size > MAX_ARCHIVE_BYTES:
                raise ValueError("uncompressed report artifact is too large")
            return files.read(member)


def main_report(actions: Actions) -> str | None:
    for _, artifact in actions.artifacts("main", REPORT_ARTIFACT):
        html = actions.read_artifact(artifact, "index.html").decode("utf-8")
        if BRANCH_ELEMENT.search(html) and 'id="branch-select"' in html:
            return html
        # The latest report predates full branch reports. Wait for the next main run.
        return None
    return None


def branch_snapshot(actions: Actions, branch: str):
    for run, artifact in actions.artifacts(branch, BRANCH_ARTIFACT):
        try:
            data = json.loads(actions.read_artifact(artifact, BRANCH_FILE))
            view = branch_report.report_view(data, run["head_sha"])
        except (AttributeError, BadZipFile, KeyError, TypeError, ValueError) as exc:
            print(f"Skipping invalid report artifact {artifact['id']} for {branch!r}: {exc}")
            continue
        return {
            "artifact": artifact["id"],
            "commit": run["head_sha"],
            "generated": artifact["created_at"],
        }, {**view, "revision": {"ref": branch, "commit": run["head_sha"], "generated": artifact["created_at"]}}
    return None


def assemble(actions: Actions, output: Path) -> bool:
    html = main_report(actions)
    if html is None:
        return False

    branch_dir = output / "branches"
    branch_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for branch in sorted(actions.branches(), key=lambda item: item["name"].casefold()):
        name = branch["name"]
        if name == "main" or name in GENERATED_BRANCHES:
            continue
        entry = {"name": name, "head": branch["commit"]["sha"]}
        result = branch_snapshot(actions, name)
        if result is not None:
            metadata, view = result
            entry.update(metadata)
            artifact = metadata["artifact"]
            if type(artifact) is not int or artifact < 1:
                raise ValueError("invalid report artifact ID")
            path = branch_dir / f"{artifact}.json"
            path.write_text(json.dumps(view, separators=(",", ":")), encoding="utf-8")
        manifest.append(entry)

    payload = json.dumps(manifest, separators=(",", ":")).replace("<", "\\u003c")
    html = BRANCH_ELEMENT.sub(lambda match: match[1] + payload + match[2], html, count=1)
    (output / "index.html").write_text(html, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    if not assemble(Actions(args.repository), args.output):
        print("Waiting for a successful main run of Build & Tests with the branch-capable report.")
        return
    with args.github_output.open("a", encoding="utf-8") as output:
        output.write("published=true\n")


if __name__ == "__main__":
    main()
