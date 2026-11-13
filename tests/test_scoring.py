"""Unit tests for the accuracy scoring (eval_task) and the paired significance test (report).

Run:  .venv/bin/python -m unittest discover -s tests -v
No GPU, server or dataset download needed.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_task import clean_sql, exec_match, f1, norm  # noqa: E402
from report import mcnemar_exact, pareto  # noqa: E402

SCHEMA = """
CREATE TABLE robots (id INT, name TEXT, battery INT);
INSERT INTO robots VALUES (1, 'arm', 80), (2, 'rover', 35), (3, 'drone', 60);
"""


class CleanAndNormalise(unittest.TestCase):
    def test_strips_markdown_fence_and_semicolon(self):
        self.assertEqual(clean_sql("```sql\nSELECT 1;\n```"), "SELECT 1")

    def test_strips_bare_fence(self):
        self.assertEqual(clean_sql("```\nSELECT 1\n```"), "SELECT 1")

    def test_norm_ignores_case_and_whitespace(self):
        self.assertEqual(norm("SELECT  name\nFROM robots;"), norm("select name from robots"))

    def test_norm_keeps_real_differences(self):
        self.assertNotEqual(norm("SELECT name FROM robots"), norm("SELECT id FROM robots"))


class ExecMatch(unittest.TestCase):
    def test_same_query_matches(self):
        q = "SELECT name FROM robots WHERE battery > 50"
        self.assertTrue(exec_match(SCHEMA, q, q))

    def test_different_text_same_result_matches(self):
        gold = "SELECT name FROM robots WHERE battery > 50"
        pred = "SELECT r.name FROM robots AS r WHERE r.battery >= 51"
        self.assertTrue(exec_match(SCHEMA, gold, pred))

    def test_row_order_does_not_matter(self):
        gold = "SELECT name FROM robots ORDER BY id"
        pred = "SELECT name FROM robots ORDER BY id DESC"
        self.assertTrue(exec_match(SCHEMA, gold, pred))

    def test_wrong_result_does_not_match(self):
        gold = "SELECT name FROM robots WHERE battery > 50"
        pred = "SELECT name FROM robots WHERE battery < 50"
        self.assertFalse(exec_match(SCHEMA, gold, pred))

    def test_invalid_prediction_counts_as_wrong(self):
        self.assertFalse(exec_match(SCHEMA, "SELECT name FROM robots", "SELEC name FROM"))

    def test_unrunnable_reference_is_skipped_not_scored(self):
        # None means "leave this question out of exec match", which is why exec_scored < n
        self.assertIsNone(exec_match(SCHEMA, "SELECT * FROM no_such_table", "SELECT 1"))

    def test_fenced_prediction_is_cleaned_before_running(self):
        gold = "SELECT COUNT(*) FROM robots"
        self.assertTrue(exec_match(SCHEMA, gold, "```sql\nSELECT COUNT(*) FROM robots;\n```"))


class TokenF1(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(f1("a b c", "a b c"), 1.0)

    def test_disjoint(self):
        self.assertEqual(f1("a b", "c d"), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(f1("a b c d", "a b"), 2 * 0.5 * 1.0 / 1.5)


class McNemar(unittest.TestCase):
    def test_identical_outcomes_give_p_one(self):
        a = [1, 0, 1, 1, 0]
        self.assertEqual(mcnemar_exact(a, a), (0, 0, 1.0))

    def test_counts_discordant_pairs_and_skips_none(self):
        a = [1, 1, 0, None, 1]
        b = [0, 1, 1, 1, None]
        only_a, only_b, _ = mcnemar_exact(a, b)
        self.assertEqual((only_a, only_b), (1, 1))

    def test_known_value_10_vs_0(self):
        # 10 discordant pairs all one way: p = 2 * 0.5**10
        a, b = [1] * 10, [0] * 10
        self.assertAlmostEqual(mcnemar_exact(a, b)[2], 2 / 1024)

    def test_known_value_3_vs_7(self):
        # two-sided exact binomial, n=10, k=3: 2 * (1+10+45+120)/1024 = 0.34375
        a = [1] * 3 + [0] * 7
        b = [0] * 3 + [1] * 7
        self.assertAlmostEqual(mcnemar_exact(a, b)[2], 0.34375)

    def test_is_symmetric(self):
        a = [1, 1, 1, 0, 0, 1, 0]
        b = [0, 1, 0, 0, 1, 0, 0]
        self.assertAlmostEqual(mcnemar_exact(a, b)[2], mcnemar_exact(b, a)[2])


class Pareto(unittest.TestCase):
    def test_dominated_point_is_dropped(self):
        pts = [{"speed": 10, "acc": 0.5}, {"speed": 20, "acc": 0.6}, {"speed": 5, "acc": 0.9}]
        front = pareto(pts)
        self.assertEqual([(p["speed"], p["acc"]) for p in front], [(5, 0.9), (20, 0.6)])


if __name__ == "__main__":
    unittest.main()
