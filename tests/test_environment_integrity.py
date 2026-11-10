"""Regression checks for raw missing states, time isolation and Windows paths."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from sems_tools import CaseStore, MarketEnvironment
from sems.evaluation.review_events import evaluate_review_events

T0 = 1600000000000
DAY = 86400000


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        files = {
            "case/case.json": {
                "case_id": "audit",
                "t0_timestamp_ms": T0,
                "review_window_end_timestamp_ms": T0 + 3 * DAY,
            },
            "case/series.json": {"series": {}},
            "case/product.json": {"keepa": {"static": {}}},
        }
        for relative, value in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        (self.root / "case/shelf.jsonl").write_text(
            json.dumps(
                {
                    "product_id": "A",
                    "title": "A",
                    "product_record_path": "product.json",
                    "smoothed_history_path": "series.json",
                    "product_reviews_path": "reviews.jsonl",
                }
            ),
            encoding="utf-8",
        )
        (self.root / "case/reviews.jsonl").write_text("", encoding="utf-8")
        profiles = self.root / "case/user_profiles"
        profiles.mkdir()
        (profiles / "user.jsonl").write_text(
            json.dumps({"review_id": "old", "user_id": "user", "timestamp_ms": T0 - DAY}),
            encoding="utf-8",
        )
        self.market = MarketEnvironment(self.root, {"simulation_mode": True, "query_budget": 20})
        self.addCleanup(self.market.close)

    def test_directory_loads_profiles_on_windows(self):
        self.assertEqual(self.market.data.asins, ("A",))
        self.assertEqual(set(self.market.data.users), {"user"})

    def test_smoothed_missing_price_does_not_resurrect_previous_offer(self):
        m = self.market
        m.data.case["smoothed_values_may_use_future_endpoints"] = True
        m.data.series["A"]["csv:NEW"] = [
            {"timestamp_ms": T0 - 2000, "value": 1000, "raw_value": 1000, "status": "observed"},
            {"timestamp_ms": T0 - 1000, "value": 1500, "raw_value": -1, "status": "interpolated"},
            {"timestamp_ms": T0 + DAY, "value": 2000, "raw_value": 2000, "status": "observed"},
        ]
        q = m.open_session("user")
        self.assertIsNone(
            q.call("get_price_info", asin="A", view="offers")["data"]["offers"]["new"]["value"]
        )
        history = q.call("get_price_info", asin="A", view="history", window_days=30)["data"][
            "observations"
        ]
        self.assertEqual([r["price"]["value"] for r in history], [10, None])
        self.assertEqual(m._value("A", "csv:NEW")["state"], "unavailable")
        m.commit_day(T0 + DAY)
        self.assertEqual(m._value("A", "csv:NEW")["value"], 20)

    def test_future_rank_rating_count_remain_at_t0(self):
        m = self.market
        for key, initial, future in [
            ("csv:SALES", 10, 1),
            ("csv:RATING", 4, 1),
            ("csv:COUNT_REVIEWS", 100, 200),
        ]:
            m.data.series["A"][key] = [
                {"timestamp_ms": T0, "value": initial, "status": "observed"},
                {"timestamp_ms": T0 + DAY, "value": future, "status": "observed"},
            ]
        m.commit_day(T0 + DAY)
        self.assertEqual(m._value("A", "csv:SALES")["value"], 10)
        self.assertEqual(m._value("A", "csv:RATING")["value"], 4)
        self.assertEqual(m._value("A", "csv:COUNT_REVIEWS")["value"], 100)

    def test_simulated_writes_only_become_visible_after_commit(self):
        m = self.market
        s = m.open_session("user", "source")
        other = m.open_session("other")
        s.call("purchase_product", asin="A")
        s.call("rate_product", asin="A", rating=5)
        s.call("publish_review", asin="A", text="simulated review")
        self.assertEqual(
            other.call("get_rank_info", asin="A")["data"]["simulated_purchase_count"], 0
        )
        self.assertEqual(
            other.call("get_reviews", asin="A", sort="recent", rating_filter="all")["data"][
                "reviews"
            ],
            [],
        )
        m.commit_day(T0 + DAY)
        self.assertEqual(
            other.call("get_rank_info", asin="A")["data"]["simulated_purchase_count"], 1
        )
        self.assertEqual(
            other.call("get_reviews", asin="A", sort="recent", rating_filter="all")["data"][
                "reviews"
            ][0]["text"],
            "simulated review",
        )

    def test_incomplete_truth_universe_cannot_receive_complete_metrics(self):
        row = {
            "session_id": "s",
            "user_id": "u",
            "scheduled_at_ms": 1,
            "status": "ok",
            "decision": {
                "product_id": "A",
                "rating_probabilities": [0, 0, 0, 0, 1],
                "rating": 5,
                "review_text": "ok",
            },
        }
        truth = {
            "s": {"user_id": "u", "timestamp_ms": 1, "asin": "A", "rating": 5},
            "missing": {"user_id": "v", "timestamp_ms": 2, "asin": "A", "rating": 1},
        }
        result = evaluate_review_events({"days": [{"sessions": [row]}]}, truth, ["A"])
        self.assertIsNone(result["metrics"]["JSD_full"]["value"])
        self.assertEqual(result["coverage"]["missing_simulation_events"], 1)
        self.assertEqual(result["coverage"]["truth_events"], 2)

    def test_future_real_review_is_never_returned(self):
        m = self.market
        m.data.reviews["A"] = [
            {
                "review_id": "past",
                "timestamp_ms": T0 - 1000,
                "review_timestamp_utc": datetime.fromtimestamp(
                    (T0 - 1000) / 1000, timezone.utc
                ).isoformat(),
                "text": "past",
                "rating": 4,
            },
            {
                "review_id": "future",
                "timestamp_ms": T0 + 1000,
                "review_timestamp_utc": datetime.fromtimestamp(
                    (T0 + 1000) / 1000, timezone.utc
                ).isoformat(),
                "text": "future",
                "rating": 5,
            },
        ]
        m.commit_day(T0 + DAY)
        result = m.open_session("reader").call(
            "get_reviews", asin="A", sort="recent", rating_filter="all"
        )
        self.assertEqual([r["id"] for r in result["data"]["reviews"]], ["past"])


if __name__ == "__main__":
    unittest.main()
