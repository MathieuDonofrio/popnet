import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import code_metrics


SCRIPT = Path(__file__).with_name("code_metrics.py")


class CodeMetricsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()

    def write(self, relative_path, source):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=self.root, capture_output=True, text=True,
        )

    def output(self, *args):
        result = self.cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return result.stdout

    def assert_cli_error(self, message, *args):
        result = self.cli(*args)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(message, result.stderr)

    def test_both_metrics_share_recursive_deduplicated_scope(self):
        source = "int f() { return 1; }\n"
        first = self.write("src/nested dir/one.CPP", source)
        second = self.write("include/two.HH", source)
        empty = self.write("include/empty.h", "")
        self.write("src/nested dir/ignore.txt", source)
        self.write("tests/outside.cpp", source)
        directories = ["src", "include", "src/nested dir", "src"]

        files = code_metrics.source_files([self.root / directory for directory in directories])
        sloc = json.loads(self.output("sloc", *directories, "--json"))
        cognitive = json.loads(self.output("cognitive", *directories, "--json"))

        self.assertEqual(files, sorted([first, second, empty]))
        self.assertEqual(sloc, {
            "schema_version": 1,
            "metric": "sloc",
            "directories": [str(self.root / directory) for directory in directories],
            "files": 3,
            "value": 2,
        })
        self.assertEqual(cognitive, {
            **sloc, "metric": "cognitive", "functions": 2, "percentile": 95, "value": 0,
        })

    def test_sloc_excludes_blank_and_comment_only_lines(self):
        self.write("src/code.c", """
// comment
/* block
   comment */
int value = 0; // trailing comment
const char *text = "/*"; /* trailing block */
int another = 0;
""")

        self.assertEqual(self.output("sloc", "src"), "3\n")

    def test_sloc_all_empty_files(self):
        self.write("src/empty.h", "")

        self.assertEqual(self.output("sloc", "src"), "0\n")

    def test_default_and_custom_nearest_rank(self):
        self.write("src/functions.cpp", "\n".join(
            [f"int f{i}() {{ return {i}; }}" for i in range(18)] + [
                "int one(int x) { if (x) return 1; return 0; }",
                "int two(int x) { if (x && x > 2) return 1; return 0; }",
            ],
        ))

        self.assertEqual(self.output("cognitive", "src"), "1\n")
        for percentile, expected in [("0.01", 0), ("90", 0), ("90.01", 1), ("100", 2)]:
            with self.subTest(percentile=percentile):
                self.assertEqual(self.output("cognitive", "src", "--percentile", percentile), f"{expected}\n")

    def test_cognitive_complexity_counts_switch_once_and_weights_nesting(self):
        self.write("src/functions.cpp", """
int dispatch(int x) {
    switch (x) {
        case 0: return 1;
        case 1: return 2;
        case 2: return 3;
        default: return 0;
    }
}
int nested(int x, int y) {
    if (x) {
        if (y) return 1;
    }
    return 0;
}
""")

        self.assertEqual(self.output("cognitive", "src", "--percentile", "50"), "1\n")
        self.assertEqual(self.output("cognitive", "src"), "3\n")

    def test_shields_badges(self):
        self.write("src/code.cpp", "int f() { return 0; }\n" + "int x;\n" * 1249)

        for metric, flags, label, message, color in [
            ("sloc", [], "Source Lines of Code", "1,250", "blue"),
            ("cognitive", [], "P95 Cognitive Complexity", "0", "#26bf26"),
            ("cognitive", ["--percentile", "90.5"], "P90.5 Cognitive Complexity", "0", "#26bf26"),
        ]:
            with self.subTest(metric=metric, flags=flags):
                badge = json.loads(self.output(metric, "src", "--badge", *flags))
                self.assertEqual(badge, {
                    "schemaVersion": 1, "label": label, "message": message, "color": color,
                })

    def test_complexity_badge_color_gradient(self):
        for score, color in [(0, "#26bf26"), (15, "#bfbf26"), (30, "#bf2626"), (40, "#bf2626")]:
            with self.subTest(score=score):
                self.assertEqual(code_metrics.make_badge("cognitive", score, 95)["color"], color)

    def test_invalid_percentiles(self):
        for percentile in ["0", "-1", "100.01", "nan", "inf", "wrong"]:
            with self.subTest(percentile=percentile):
                self.assert_cli_error("percentile must be", "cognitive", "src", "--percentile", percentile)

        self.assert_cli_error("--percentile is only valid for cognitive", "sloc", "src", "--percentile", "95")

    def test_mutually_exclusive_output_modes(self):
        self.assert_cli_error("not allowed with argument", "cognitive", "src", "--json", "--badge")

    def test_invalid_or_empty_directories_and_no_functions(self):
        self.write("empty/notes.txt", "no code")
        self.write("src/declaration.h", "int f();\n")

        self.assert_cli_error("Not a directory", "sloc", "missing")
        for metric in ["sloc", "cognitive"]:
            self.assert_cli_error("No C/C++ source files found", metric, "empty")
        self.assert_cli_error("No C/C++ functions found", "cognitive", "src")

    def test_cloc_partial_report_is_an_error(self):
        file = self.write("src/source.c", "int value;\n")
        report = subprocess.CompletedProcess([], 0, '{"SUM": {"nFiles": 0, "code": 0}}', "")

        with patch("code_metrics.subprocess.run", return_value=report):
            with self.assertRaisesRegex(RuntimeError, "cloc counted 0 of 1 nonempty files"):
                code_metrics.count_sloc([file])

    def test_lizard_parser_diagnostic_is_an_error(self):
        file = self.write("src/source.h", "int f() { return 0; }\n")

        def fail_to_parse(*_):
            print("[skip] fail to process", file=sys.stderr)

        with patch("lizard.FileAnalyzer.analyze_source_code", side_effect=fail_to_parse):
            with self.assertRaisesRegex(RuntimeError, "Lizard failed to analyze .*source.h"):
                code_metrics.count_cognitive_complexity([file], 95)
