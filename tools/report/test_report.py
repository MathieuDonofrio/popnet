import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch
from urllib.parse import urlunsplit

import generate
from tools.report import coverage_data
from tools.report import markdown_preview


class ReportHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_data = False
        self.payload = []
        self.scripts = 0
        self.external_resources = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.scripts += 1
            self.in_data = attrs.get("id") == "report-data" and attrs.get("type") == "application/json"
        if "src" in attrs or (tag == "link" and "href" in attrs):
            self.external_resources.append(attrs)

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.payload.append(data)


class ReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text).lstrip("\n"), encoding="utf-8")
        return path

    def write_trace_fixture(self):
        self.write("docs/intents/INT-0001-usable-tensors.md", """
            ---
            id: INT-0001
            title: Usable tensors
            ---
            ## Problem
            Tensor behavior must be predictable.
            ## Desired outcome
            Tensor storage remains usable.
            ### Retention
            Keep **all** referenced storage.
            ## Scope
            Tensor storage.
            ## Success assessment
            Retain and inspect a tensor.
            ## Maintenance notes
            Document assumptions about how retained views are used.
        """)
        self.write("docs/requirements/REQ-0001-storage.md", """
            ---
            id: REQ-0001
            title: Stable storage
            intents: [INT-0001]
            specifications: [docs/specifications/tensor.md#retention]
            ---
            ## Obligation
            Retain storage.
            ### Lifetime
            Store `data` until the last reference is released.
            ## Rationale
            Avoid invalidating active views.
        """)
        self.write("docs/requirements/REQ-0002-layout.md", """
            ---
            id: REQ-0002
            title: Stable layout
            intents: [INT-0001]
            specifications: []
            ---
            ## Obligation
            Retain layout.
        """)
        self.write("docs/requirements/REQ-0003-view.md", """
            ---
            id: REQ-0003
            title: Tensor view
            intents: []
            specifications: []
            ---
            ## Obligation
            Preserve views.
        """)
        self.write("docs/specifications/tensor.md", "# Tensor\n## Retention\nRetain storage.\n")
        self.write("tests/tensor.cpp", """
            // @verifies REQ-0001
            TEST_CASE("retain storage") {}
        """)

    def test_lcov_maps_paths_and_reads_branch_outcomes(self):
        source = self.write("src/cpu/example.cpp", "one\ntwo\nthree\n")
        lcov = self.write("coverage.info", f"""
            SF:{source}
            DA:2,7
            BRDA:1,0,0,0
            BRDA:1,0,1,-
            BRDA:3,1,2,0
            end_of_record
            SF:src\\cpu\\example.cpp
            BRDA:1,0,0,2
            end_of_record
            SF:tests/test.cpp
            BRDA:1,0,0,1
            end_of_record
        """)

        coverage = coverage_data.read_lcov(lcov, self.root, {"src/cpu/example.cpp": ["one", "two", "three"]})
        lines = coverage["branches"]["src/cpu/example.cpp"]

        self.assertEqual(coverage_data.total(lines[1]), {"covered": 1, "total": 2})
        self.assertEqual(coverage_data.total(lines[3]), {"covered": 0, "total": 1})
        self.assertNotIn(2, lines)
        self.assertEqual(coverage["lines"]["src/cpu/example.cpp"][2][0]["outcomes"], {"Executed": True})

    def test_json_merges_instantiations_and_distinguishes_conditions_on_the_same_line(self):
        path = "src/example.cpp"
        source = ["if (x && y) {}", "if (z) {}", "MACRO();"]
        lcov_json = self.write("coverage.json", json.dumps({
            "type": "llvm.coverage.json.export",
            "data": [{"files": [{
                "filename": str(self.root / path),
                "branches": [
                    [1, 5, 1, 6, 18, 0, 0, 0, 6],
                    [1, 5, 1, 6, 0, 1, 0, 0, 6],
                    [1, 10, 1, 11, 5, 0, 0, 0, 6],
                ],
                "expansions": [{
                    "source_region": [3, 1, 3, 6, 1, 0, 1, 1],
                    "filenames": [path, "include/macros.h"],
                    "branches": [[5, 2, 5, 6, 0, 0, 1, 0, 6]],
                }],
            }]}],
        }))

        coverage = coverage_data.read_json(lcov_json, self.root, {path: source})
        branches = coverage[path]

        self.assertEqual(coverage_data.total(branches[1]), {"covered": 3, "total": 4})
        self.assertEqual(branches[1][1], {"label": "Column 10 · y", "outcomes": {"True": True, "False": False}})
        self.assertNotIn(2, branches)
        self.assertIn("macro branch at macros.h:5:2", branches[3][0]["label"])

    def test_rejects_missing_or_malformed_coverage(self):
        with self.assertRaises(FileNotFoundError):
            coverage_data.read_lcov(self.root / "missing.info", self.root, {})

        for content, error in [
            ("SF:src/example.cpp\nBRDA:1,0,0,2\n", "missing end_of_record"),
            ("SF:src/example.cpp\nBRDA:1,0,0,-1\nend_of_record\n", "invalid or duplicate"),
            ("SF:src/example.cpp\nBRDA:1,0,0,oops\nend_of_record\n", "invalid literal"),
            ("BRDA:1,0,0,2\nend_of_record\n", "unexpected LCOV"),
            ("SF:tests/example.cpp\nBRDA:1,0,0,2\nend_of_record\n", "no production coverage"),
        ]:
            with self.subTest(content=content):
                lcov = self.write("coverage.info", content)
                with self.assertRaisesRegex(ValueError, error):
                    coverage_data.read_lcov(lcov, self.root, {"src/example.cpp": ["line"]})

        malformed = self.write("coverage.json", '{"type":"llvm.coverage.json.export","data":null}')
        with self.assertRaisesRegex(ValueError, "invalid LLVM coverage JSON"):
            coverage_data.read_json(malformed, self.root, {})

    def test_trace_preserves_declared_relationships_and_gaps(self):
        self.write_trace_fixture()

        trace = generate.collect_traceability(self.root)

        self.assertEqual(trace["summary"]["requirements_with_tests"], 1)
        self.assertEqual(trace["combined"], {"covered": 2, "total": 4})
        self.assertEqual(trace["intents"][0]["requirements"], ["REQ-0001", "REQ-0002"])
        intent = trace["intents"][0]["markdown"]
        self.assertIn("Tensor storage remains usable.", intent["text"])
        self.assertIn("<h2>Problem</h2>", intent["html"])
        self.assertIn("<h2>Scope</h2>", intent["html"])
        self.assertIn("<h2>Success assessment</h2>", intent["html"])
        self.assertIn("<h2>Maintenance notes</h2>", intent["html"])
        self.assertIn("<h3>Retention</h3>", intent["html"])
        self.assertIn("<strong>all</strong>", intent["html"])
        self.assertNotIn("id: INT-0001", intent["text"])
        self.assertEqual(trace["unassigned"], ["REQ-0003"])
        linked = trace["requirements"]["REQ-0001"]
        self.assertEqual(linked["intents"][0]["id"], "INT-0001")
        self.assertEqual(linked["specifications"], [{"path": "docs/specifications/tensor.md#retention"}])
        self.assertIn("Retain storage.", linked["markdown"]["text"])
        self.assertIn("<h3>Lifetime</h3>", linked["markdown"]["html"])
        self.assertIn("<h2>Rationale</h2>", linked["markdown"]["html"])
        self.assertIn("<code>data</code>", linked["markdown"]["html"])
        self.assertNotIn("specifications:", linked["markdown"]["text"])
        self.assertEqual(trace["specifications"]["docs/specifications/tensor.md#retention"], {
            "title": "Retention", "text": "Retain storage.", "html": "<p>Retain storage.</p>\n",
        })
        self.assertEqual(linked["tests"], [{"path": "tests/tensor.cpp", "line": 2, "name": "retain storage"}])
        self.assertEqual(trace["requirements"]["REQ-0002"]["tests"], [])

    def test_rejects_invalid_trace_metadata(self):
        self.write_trace_fixture()
        self.write("docs/requirements/REQ-0001-storage.md", "---\ninvalid: [\n---\n")

        with self.assertRaisesRegex(ValueError, "REQ-0001-storage.md"):
            generate.collect_traceability(self.root)

    def test_report_uses_only_production_and_linked_test_sources(self):
        self.write_trace_fixture()
        self.write("src/logic.cpp", "int decide(int x) { if (x) return 1; return 0; }\n")
        self.write("include/api.h", "int decide(int x);\n")
        self.write("src/notes.txt", "ignore")
        self.write("tests/unlinked.cpp", "// No declared tests in this file")
        self.write("bench/bench.cpp", "int bench() { return 2; }")
        lcov = self.write("coverage.info", "SF:src/logic.cpp\nDA:1,3\nBRDA:1,0,0,3\nBRDA:1,0,1,0\nend_of_record\n")
        llvm_json = self.write("coverage.json", json.dumps({
            "type": "llvm.coverage.json.export",
            "data": [{"files": [{
                "filename": "src/logic.cpp",
                "branches": [[1, 24, 1, 25, 3, 0, 0, 0, 4]],
            }]}],
        }))

        report = generate.collect_report(self.root, lcov, llvm_json)

        self.assertEqual(report["production"], ["include/api.h", "src/logic.cpp"])
        self.assertEqual(set(report["sources"]), {"include/api.h", "src/logic.cpp", "tests/tensor.cpp"})
        self.assertNotIn("coverage", report["sources"]["include/api.h"])
        self.assertEqual(report["sources"]["tests/tensor.cpp"]["lines"][0], "// @verifies REQ-0001")
        self.assertEqual(report["coverage"], {
            "branches": {"covered": 1, "total": 2},
            "lines": {"covered": 1, "total": 1},
            "combined": {"covered": 2, "total": 3},
        })
        self.assertEqual(report["sources"]["src/logic.cpp"]["coverage"]["combined"], {"covered": 2, "total": 3})
        branch = report["sources"]["src/logic.cpp"]["coverage"]["branches"]["lines"][1][0]
        self.assertEqual(branch["outcomes"], {"True": True, "False": False})
        self.assertEqual(report["sloc"], 2)
        self.assertEqual(report["functions"], [{
            "name": "decide", "file": "src/logic.cpp", "start_line": 1,
            "end_line": 1, "cognitive_complexity": 1,
        }])
        self.assertEqual(report["p95"], 1)

        badges = generate.metric_badges(report)
        self.assertEqual(badges["coverage"]["label"], "Code Coverage")
        self.assertEqual(badges["traceability"]["label"], "Requirement Coverage")
        self.assertEqual({name: badge["message"] for name, badge in badges.items()}, {
            "coverage": "66.7%",
            "traceability": "50.0%",
            "sloc": "2",
            "cognitive-complexity": "1",
        })

    def test_rejects_coverage_line_outside_current_source(self):
        self.write("src/logic.cpp", "int decide(int x);\n")
        lcov = self.write("coverage.info", "SF:src/logic.cpp\nBRDA:3,0,0,1\nend_of_record\n")

        with self.assertRaisesRegex(ValueError, "coverage line outside src/logic.cpp"):
            generate.collect_report(self.root, lcov)

    def test_specification_excerpt_keeps_nested_sections_and_skips_fenced_headings(self):
        spec = "# API\n## Behavior\nDetail.\n```\n## Example\n```\n### Notes\nMore.\n## Other\nExcluded."

        preview = generate.document_preview(spec, "behavior")

        self.assertEqual(preview["title"], "Behavior")
        self.assertIn("### Notes\nMore.", preview["text"])
        self.assertIn("## Example", preview["text"])
        self.assertNotIn("Excluded.", preview["text"])

    def test_markdown_specification_has_safe_html_and_raw_source(self):
        path = self.write("docs/specifications/example.md", """
            # Example

            | Input | Result |
            | --- | --- |
            | `x` | **yes** |

            <script>alert(1)</script>

            [unsafe](javascript:alert%281%29) ![image](image.png)
        """)

        preview = generate.specification_preview(self.root, "docs/specifications/example.md")

        self.assertEqual(preview["text"], path.read_text(encoding="utf-8").strip())
        self.assertIn("<table>", preview["html"])
        self.assertIn("<strong>yes</strong>", preview["html"])
        self.assertIn("<code>x</code>", preview["html"])
        self.assertNotIn("<script", preview["html"])
        self.assertNotIn('href="javascript:', preview["html"])
        self.assertNotIn("<img", preview["html"])

    def test_markdown_renders_inline_and_display_math_and_preserves_raw_text(self):
        source = r"Inline $x^2$, display $$y^2$$, and literal `$z$`." + "\n\n$$\n" + r"\frac{a}{b}" + "\n$$\n"
        self.write("docs/specifications/math.md", source)

        preview = generate.specification_preview(self.root, "docs/specifications/math.md")

        self.assertEqual(preview["text"], source.strip())
        self.assertEqual(preview["html"].count('<math display="inline"'), 1)
        self.assertEqual(preview["html"].count('<math display="block"'), 2)
        self.assertIn('<span class="math display">', preview["html"])
        self.assertIn("<mfrac>", preview["html"])
        self.assertIn("<code>$z$</code>", preview["html"])

    def test_math_falls_back_for_invalid_tex_and_filters_unsafe_markup(self):
        options = {"display_mode": False}
        self.assertEqual(markdown_preview.render_math(r"\frac{", options), r"<code>$\frac{$</code>")

        unsafe = f'<math xmlns="{markdown_preview.MATHML}"><mi onclick="bad()">x</mi></math>'
        with patch.object(markdown_preview, "convert", return_value=unsafe):
            self.assertNotIn("onclick", markdown_preview.render_math("x", options))
        unsafe = f'<math xmlns="{markdown_preview.MATHML}"><script>bad()</script></math>'
        with patch.object(markdown_preview, "convert", return_value=unsafe):
            self.assertEqual(markdown_preview.render_math("x", options), "<code>$x$</code>")

    def test_repository_document_base_uses_origin_and_omits_credentials(self):
        host = "github.com"
        repo = "/some-owner/some-repo"
        https_remote = urlunsplit(("https", "user:secret@" + host, repo + ".git", "", ""))
        expected = urlunsplit(("https", host, repo + "/blob/", "", ""))

        self.assertEqual(generate.repository_browse_base(https_remote), expected)
        self.assertEqual(generate.repository_browse_base(f"git@{host}:{repo.lstrip('/')}.git"), expected)
        self.assertIsNone(generate.repository_browse_base(""))
        self.assertIsNone(generate.repository_browse_base(str(self.root / "local.git")))

    def test_safe_self_contained_html_round_trip(self):
        arbitrary = '</script><img src=x onerror=alert(1)>&"\u2028'
        data = {"sources": {"src/file.cpp": {"lines": [arbitrary]}}, "title": arbitrary}

        html = generate.render_report(data)
        parser = ReportHTML()
        parser.feed(html)
        parser.close()

        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertEqual(parser.scripts, 2)
        self.assertEqual(parser.external_resources, [])
        self.assertNotIn(arbitrary, html)
        self.assertEqual(json.loads("".join(parser.payload)), data)
