import concurrent.futures
import tempfile
import unittest
from pathlib import Path
from sems.rsi.api_budget import SharedAPIBudget
from sems_tools.case_store import CaseStore


def process_attempt(path):
    try:
        return SharedAPIBudget(path, limit=6, profile_limit=4).reserve("decision", "test", False)
    except RuntimeError:
        return None


class BudgetTests(unittest.TestCase):
    def test_process_shared_quota(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "budget.sqlite")
            SharedAPIBudget(path, limit=6, profile_limit=4)
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(process_attempt, [path] * 12))
            self.assertEqual(sum(x is not None for x in results), 6)
            self.assertEqual(
                SharedAPIBudget(path, limit=6, profile_limit=4).summary()["actual_requests"], 6
            )

    def test_shared_quota_including_concurrency(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "budget.sqlite"
            a = SharedAPIBudget(path, limit=10, profile_limit=4)
            b = SharedAPIBudget(path, limit=10, profile_limit=4)

            def attempt(i):
                try:
                    return (a if i % 2 else b).reserve(
                        "profile" if i < 10 else "decision", "test", i % 3 == 0
                    )
                except RuntimeError:
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                out = list(pool.map(attempt, range(30)))
            self.assertEqual(len([v for v in out if v is not None]), 10)
            self.assertLessEqual(a.summary()["profile_requests"], 4)
            with self.assertRaises(RuntimeError):
                b.reserve("decision", "test", False)

    def test_reject_prohibited_role(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                SharedAPIBudget(Path(d) / "b").reserve("proposer", "test", False)

    def test_path_resolution(self):
        store = object.__new__(CaseStore)
        store.prefix = "cases/x/"
        store._names = {"cases/x/product_reviews/a"}
        self.assertEqual(store.resolve("product_reviews/a"), "cases/x/product_reviews/a")
        store._names.add("product_reviews/a")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            store.resolve("product_reviews/a")
        with self.assertRaisesRegex(ValueError, "missing"):
            store.resolve("missing")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            store.resolve("../a")


if __name__ == "__main__":
    unittest.main()
