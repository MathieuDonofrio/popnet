import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import generate
from tools.report import branch_coverage, branch_report, publish_site


class PublishSiteTests(unittest.TestCase):
    def coverage_snapshot(self):
        return {
            "schema_version": branch_coverage.SCHEMA_VERSION,
            "commit": "branch-commit",
            "sources": {
                "src/example.cpp": {
                    "lines": ["int value;", "// second line"],
                    "coverage": {
                        "lines": {"1": [{"label": "", "outcomes": {"Executed": True}}]},
                        "branches": {"1": [{"label": "Condition", "outcomes": {"True": True, "False": False}}]},
                    },
                },
            },
        }

    def test_publisher_rebuilds_the_combined_counts_from_branch_data(self):
        report = branch_coverage.report_view(self.coverage_snapshot(), "branch-commit")

        self.assertEqual(report["production"], ["src/example.cpp"])
        self.assertEqual(report["coverage"]["combined"], {"covered": 2, "total": 3})
        self.assertEqual(report["sources"]["src/example.cpp"]["coverage"]["combined"], {"covered": 2, "total": 3})

    def test_publisher_rejects_unrelated_commits_and_paths(self):
        snapshot = self.coverage_snapshot()

        with self.assertRaisesRegex(ValueError, "commit does not match"):
            branch_coverage.report_view(snapshot, "another-commit")
        snapshot["sources"]["src/../secret.txt"] = snapshot["sources"].pop("src/example.cpp")
        with self.assertRaisesRegex(ValueError, "invalid production path"):
            branch_coverage.report_view(snapshot, "branch-commit")

    def test_only_same_repository_successful_branch_runs_supply_artifacts(self):
        class Fixture(publish_site.Actions):
            def pages(self, path, key=None, **query):
                for index in range(1, 7):
                    yield {
                        "id": index, "name": publish_site.BRANCH_ARTIFACT, "expired": index == 5,
                        "created_at": f"2026-11-{index:02}T10:00:00Z",
                        "workflow_run": {"id": index, "repository_id": 1, "head_repository_id": 1,
                                         "head_branch": "feature" if index != 6 else "unrelated"},
                    }

            def get(self, path, **query):
                run_id = int(path.rsplit("/", 1)[1])
                self.run_requests.append(run_id)
                run = {
                    "id": run_id, "head_branch": "feature", "conclusion": "success", "event": "push",
                    "head_repository": {"full_name": self.repository}, "path": ".github/workflows/ci.yml",
                }
                if run_id == 2:
                    run["event"] = "pull_request"
                if run_id == 3:
                    run["head_repository"] = {"full_name": "someone/fork"}
                if run_id == 4:
                    run["path"] = ".github/workflows/unrelated.yml"
                return json.dumps(run)

        fixture = Fixture("owner/repo")
        fixture.run_requests = []
        artifacts = list(fixture.artifacts("feature", publish_site.BRANCH_ARTIFACT))

        self.assertEqual([run["id"] for run, _ in artifacts], [1])
        self.assertEqual(fixture.run_requests, [4, 3, 2, 1])

    def test_site_uses_main_html_and_a_separate_json_file_per_available_branch(self):
        main_html = generate.render_report({"revision": {"commit": "main-commit"}})
        markup = {"text": "## Branch obligation\n\n<script>unsafe</script>", "html": "<img onerror=unsafe>"}
        snapshot = {
            "code": self.coverage_snapshot(),
            "test_sources": {"tests/example.cpp": ["// branch test"]},
            "trace": {
                "intents": [{"id": "INT-0001", "title": "Branch intent",
                             "path": "docs/intents/intent.md", "markdown": markup}],
                "requirements": {"REQ-0001": {
                    "id": "REQ-0001", "title": "Branch requirement", "path": "docs/requirements/requirement.md",
                    "markdown": markup, "intents": [{"id": "INT-0001"}],
                    "tests": [{"name": "Branch test", "path": "tests/example.cpp", "line": 1,
                               "comment": "Verifies a branch-specific behavior."}],
                    "specifications": [{"path": "docs/specifications/example.md#math"}],
                }},
                "specifications": {"docs/specifications/example.md#math": {
                    "text": "$x^2$", "title": "Math", "html": "<img onerror=unsafe>",
                }},
            },
            "functions": [{"name": "branchFunction", "file": "src/example.cpp", "start_line": 1,
                           "end_line": 2, "cognitive_complexity": 7}],
            "sloc": 2,
        }

        class Fixture:
            def artifacts(self, branch, name):
                if branch == "main":
                    yield {}, {"id": 5}
                if branch == "feature/look":
                    yield {"head_sha": "branch-commit"}, {"id": 16, "created_at": "2026-11-20T10:00:00Z"}

            def branches(self):
                return [
                    {"name": "main", "commit": {"sha": "main-commit"}},
                    {"name": "code-metrics-history", "commit": {"sha": "history"}},
                    {"name": "feature/look", "commit": {"sha": "newer-commit"}},
                    {"name": "waiting", "commit": {"sha": "waiting-commit"}},
                ]

            def read_artifact(self, artifact, filename):
                if filename == "index.html":
                    return main_html.encode()
                return json.dumps(snapshot).encode()

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.assertTrue(publish_site.assemble(Fixture(), output))
            page = (output / "index.html").read_text(encoding="utf-8")
            manifest = json.loads(page.split('<script id="report-branches" type="application/json">')[1].split("</script>")[0])
            files = sorted(path.name for path in (output / "branches").iterdir())
            branch = json.loads((output / "branches/16.json").read_text(encoding="utf-8"))

        self.assertEqual([entry["name"] for entry in manifest], ["feature/look", "waiting"])
        self.assertEqual(manifest[0]["head"], "newer-commit")
        self.assertEqual(manifest[0]["commit"], "branch-commit")
        self.assertNotIn("artifact", manifest[1])
        self.assertEqual(files, ["16.json"])
        self.assertNotIn("int value;", page)
        self.assertIn("main-commit", page)
        self.assertEqual(branch["sources"]["src/example.cpp"]["lines"][0], "int value;")
        self.assertEqual(branch["sources"]["tests/example.cpp"]["lines"], ["// branch test"])
        self.assertEqual(branch["revision"]["commit"], "branch-commit")
        self.assertEqual(branch["trace"]["combined"], {"covered": 2, "total": 2})
        self.assertEqual(branch["trace"]["intents"][0]["requirements"], ["REQ-0001"])
        self.assertEqual(branch["trace"]["requirements"]["REQ-0001"]["tests"][0]["comment"],
                         "Verifies a branch-specific behavior.")
        self.assertEqual(branch["functions"][0]["name"], "branchFunction")
        self.assertEqual(branch["p95"], 7)
        html = branch["trace"]["requirements"]["REQ-0001"]["markdown"]["html"]
        self.assertIn("<h2>Branch obligation</h2>", html)
        self.assertNotIn("<script>", html)
        self.assertNotIn("onerror", html)
        spec = branch["trace"]["specifications"]["docs/specifications/example.md#math"]["html"]
        self.assertIn("<math ", spec)
        self.assertNotIn("onerror", spec)
        exported = branch_report.snapshot(branch, "branch-commit")
        self.assertNotIn("html", json.dumps(exported))
        self.assertEqual(branch_report.report_view(json.loads(json.dumps(exported)), "branch-commit")["trace"],
                         branch["trace"])

    def test_archive_reader_only_reads_the_expected_member(self):
        archive = BytesIO()
        with ZipFile(archive, "w") as files:
            files.writestr(publish_site.BRANCH_FILE, "{}")
            files.writestr("../index.html", "untrusted html")

        class Fixture(publish_site.Actions):
            def get(self, path, **query):
                return archive.getvalue()

        fixture = Fixture("owner/repo")
        self.assertEqual(fixture.read_artifact({"id": 1, "size_in_bytes": len(archive.getvalue())},
                                               publish_site.BRANCH_FILE), b"{}")
