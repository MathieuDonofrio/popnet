import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent


SCRIPT = Path(__file__).with_name("trace.py")
REQUIREMENT = "docs/requirements/REQ-0001-tensor-lifetime.md"


class TraceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.working_directory = Path(tmp.name)
        self.root = self.working_directory / "repo"
        self.script = self.root / "tools/trace/trace.py"
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)

    def write(self, relative_path, text):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text).lstrip("\n"), encoding="utf-8")

    def write_requirement(self, references):
        self.write(REQUIREMENT, (
            "---\nid: REQ-0001\ntitle: Tensor lifetime\n"
            + dedent(references).lstrip("\n")
            + "---\n## Obligation\nPreserve tensor storage while retained.\n"
        ))

    def write_repository(self):
        self.write("docs/intents/INT-0001-safe-tensors.md", """
            ---
            id: INT-0001
            title: Safe tensors
            ---
            ## Problem
            Tensor lifetime must be predictable.
            ## Desired outcome
            Retained tensors remain usable.
            ## Scope
            Tensor storage.
            ## Success assessment
            Exercise retention in a test.
        """)
        self.write_requirement("""
            intents:
              - INT-0001
            specifications:
              - docs/specifications/tensor.md#retention
        """)
        self.write("docs/requirements/REQ-0002-tensor-layout.md", """
            ---
            id: REQ-0002
            title: Tensor layout
            intents: []
            specifications: []
            ---
            ## Obligation
            Expose a stable tensor layout.
        """)
        self.write("docs/specifications/tensor.md", """
            # Tensor storage
            ## Retention
            Tensor lifetime follows reference counts.
        """)
        self.write("tests/tensor/lifetime.cpp", """
            // @verifies REQ-0001: Exercises retained storage
            TEST_CASE("retained tensor") {
                SUBCASE("retained by a view") {}
            }
        """)

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, str(self.script), *args],
            cwd=self.working_directory, capture_output=True, text=True,
        )

    def json_output(self, *args, status=0):
        result = self.cli(*args)
        self.assertEqual(result.returncode, status, result.stderr or result.stdout)
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    def test_search_finds_titles_and_specification_text(self):
        self.write_repository()

        report = self.json_output("search", "TENSOR LIFETIME", "--json")
        empty = self.json_output("search", "unmatched", "--json")

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["results"][0]["id"], "REQ-0001")
        self.assertIn("docs/specifications/tensor.md", [item["path"] for item in report["results"]])
        self.assertEqual(empty["results"], [])

    def test_show_follows_intent_requirement_specification_and_test_links(self):
        self.write_repository()

        intent = self.json_output("show", "INT-0001", "--json")
        requirement = self.json_output("show", "REQ-0001", "--json")

        self.assertEqual([item["id"] for item in intent["requirements"]], ["REQ-0001"])
        self.assertEqual([item["id"] for item in requirement["intents"]], ["INT-0001"])
        self.assertEqual(requirement["specifications"], [{"path": "docs/specifications/tensor.md#retention"}])
        tests = [{"path": "tests/tensor/lifetime.cpp", "line": 2, "name": "retained tensor",
                  "comment": "Exercises retained storage"}]
        self.assertEqual(intent["requirements"][0]["tests"], tests)
        self.assertEqual(requirement["tests"], tests)

    def test_test_link_comments_are_specific_to_the_requirement(self):
        self.write_repository()
        self.write("tests/tensor/lifetime.cpp", """
            // @verifies REQ-0001:  Retains the input: even after release.
            // @verifies REQ-0002:  Reports the expected shape.
            TEST_CASE("test for both requirements") {}
            // @verifies REQ-0001
            TEST_CASE("no comment") {}
        """)

        first = self.json_output("show", "REQ-0001", "--json")["tests"]
        second = self.json_output("show", "REQ-0002", "--json")["tests"]

        self.assertEqual(first[0]["comment"], "Retains the input: even after release.")
        self.assertNotIn("comment", first[1])
        self.assertEqual(second[0]["comment"], "Reports the expected shape.")

    def test_coverage_reports_global_gaps_and_intent_scope(self):
        self.write_repository()

        overall = self.json_output("coverage", "--json")
        scoped = self.json_output("coverage", "INT-0001", "--json")

        self.assertEqual(overall["summary"], {
            "intents_total": 1,
            "intents_with_requirements": 1,
            "requirements_total": 2,
            "requirements_with_tests": 1,
            "test_traceability_percent": 50.0,
        })
        for key in ("requirements_without_intents", "requirements_without_tests"):
            self.assertEqual([req["id"] for req in overall[key]], ["REQ-0002"])
        self.assertEqual(scoped["summary"]["requirements_total"], 1)
        self.assertEqual(scoped["summary"]["test_traceability_percent"], 100.0)
        self.assertEqual(scoped["requirements_without_tests"], [])

    def test_endpoint_badge_and_exclusive_output_flags(self):
        self.write_repository()

        badge = self.json_output("coverage", "--badge")
        conflict = self.cli("coverage", "--badge", "--json")

        self.assertEqual(badge, {
            "schemaVersion": 1, "label": "traceability", "message": "50%", "color": "yellow",
        })
        self.assertEqual(conflict.returncode, 2)
        self.assertEqual(conflict.stdout, "")
        self.assertIn("--badge and --json cannot be combined", conflict.stderr)

    def test_validate_allows_missing_links(self):
        self.write_repository()

        self.assertEqual(self.json_output("validate", "--json"), {
            "schema_version": 1, "errors": 0, "warnings": 0, "diagnostics": [],
        })

    def test_validate_rejects_invalid_document_metadata(self):
        self.write_repository()
        self.write("docs/architecture/ADR-0001-storage.md", """
            ---
            id: ADR-0001
            title: Storage
            title: Duplicate title
            ---
            ## Context
            Tensor storage lifetime.
        """)

        report = self.json_output("validate", "--json", status=1)

        self.assertIn(
            ("INVALID_YAML", "docs/architecture/ADR-0001-storage.md", 2),
            [(item["code"], item["path"], item["line"]) for item in report["diagnostics"]],
        )
        self.assertIn("MISSING_SECTION", {item["code"] for item in report["diagnostics"]})

    def test_validate_reports_duplicate_and_broken_document_references(self):
        self.write_repository()
        self.write_requirement("""
            intents:
              - INT-0001
              - INT-0001
              - INT-9999
            specifications:
              - docs/specifications/tensor.md#missing
              - docs/specifications/missing.md
        """)

        report = self.json_output("validate", "--json", status=1)

        self.assertEqual(
            [(item["code"], item["path"], item["line"]) for item in report["diagnostics"]],
            [
                ("DUPLICATE_REFERENCE", REQUIREMENT, 6),
                ("UNKNOWN_INTENT", REQUIREMENT, 7),
                ("UNKNOWN_SECTION", REQUIREMENT, 9),
                ("UNKNOWN_SPECIFICATION", REQUIREMENT, 10),
            ],
        )

    def test_validate_skips_fenced_headings_and_resolves_repeated_headings(self):
        self.write_repository()
        self.write_requirement("""
            intents: [INT-0001]
            specifications:
              - docs/specifications/tensor.md#retention-1
              - docs/specifications/tensor.md#example-only
        """)
        self.write("docs/specifications/tensor.md", """
            # Tensor storage
            ## Retention
            ~~~markdown
            ## Example only
            ~~~
            ## Retention
        """)

        report = self.json_output("validate", "--json", status=1)

        self.assertEqual(
            [(item["code"], item["line"]) for item in report["diagnostics"]],
            [("UNKNOWN_SECTION", 7)],
        )

    def test_validate_reports_bad_doctest_annotations(self):
        self.write_repository()
        self.write("tests/tensor/lifetime.cpp", """
            // @verifies REQ-0001
            // @verifies REQ-0001
            // @verifies REQ-9999
            // @verifies not-a-requirement
            TEST_CASE("bad links") {}
            TEST_CASE("no link") {}
            // @verifies REQ-0001
        """)

        report = self.json_output("validate", "--json", status=1)

        self.assertEqual(report["errors"], 5)
        self.assertEqual(
            [(item["code"], item["line"]) for item in report["diagnostics"]],
            [
                ("DUPLICATE_ANNOTATION", 2),
                ("UNKNOWN_REQUIREMENT", 3),
                ("INVALID_ANNOTATION", 4),
                ("MISSING_ANNOTATION", 6),
                ("MISPLACED_ANNOTATION", 7),
            ],
        )
        self.assertEqual({item["path"] for item in report["diagnostics"]}, {"tests/tensor/lifetime.cpp"})

    def test_unknown_ids_and_repository_load_failure(self):
        failure = self.json_output("validate", "--json", status=2)
        self.write_repository()

        missing = self.json_output("show", "REQ-9999", "--json", status=1)
        wrong_kind = self.json_output("coverage", "REQ-0001", "--json", status=1)

        self.assertEqual(failure["error"]["code"], "LOAD_FAILED")
        self.assertEqual(missing["error"]["code"], "NOT_FOUND")
        self.assertEqual(wrong_kind["error"]["code"], "NOT_FOUND")
