import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from sems.simulation_time import Clock, DAY_MS, build_schedule
from sems_tools import MarketEnvironment

CASE = Path(__import__("os").environ["SEMS_CASE_PATH"])


class SimulationInfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.market = MarketEnvironment(CASE, {"simulation_mode": True, "query_budget": 30})
        self.clock = Clock(self.market.data.t0_ms, self.market.data.end_ms)

    def tearDown(self):
        self.market.close()

    def test_clock_and_schedule(self):
        c = self.market.data
        self.assertEqual(int(c.t0.timestamp() * 1000), c.case["t0_timestamp_ms"])
        self.assertEqual(self.clock.day(0).start_ms, c.t0_ms)
        self.assertEqual(self.clock.day(len(self.clock) - 1).end_ms, c.end_ms)
        self.assertEqual(Clock(123, 123 + DAY_MS + 5).day(1).end_ms, 123 + DAY_MS + 5)
        sessions, truth = build_schedule(c)
        expected = [
            r
            for a in c.asins
            for r in c.reviews[a]
            if c.t0_ms <= r.get("timestamp_ms", 0) < c.end_ms and r.get("user_id") in c.users
        ]
        self.assertEqual(len(sessions), len(expected))
        self.assertEqual(len(truth), len(expected))
        self.assertEqual(sessions, build_schedule(c)[0])
        self.assertEqual(
            [s.scheduled_at_ms for s in sessions], sorted(s.scheduled_at_ms for s in sessions)
        )
        self.assertTrue(any(n > 1 for n in Counter(s.user_id for s in sessions).values()))
        self.assertTrue(
            all(
                set(s.to_dict()) == {"session_id", "user_id", "scheduled_at_ms", "simulation_day"}
                for s in sessions
            )
        )
        fake = SimpleNamespace(
            t0_ms=1000,
            end_ms=1000 + 3 * DAY_MS,
            asins=("a",),
            users={"u": []},
            reviews={
                "a": [
                    {"asin": "a", "review_id": "r1", "user_id": "u", "timestamp_ms": 1001},
                    {"asin": "a", "review_id": "r2", "user_id": "u", "timestamp_ms": 1002},
                    {
                        "asin": "a",
                        "review_id": "r3",
                        "user_id": "u",
                        "timestamp_ms": 1000 + 2 * DAY_MS,
                    },
                ]
            },
        )
        example, ground_truth = build_schedule(fake)
        self.assertEqual([s.simulation_day for s in example], [0, 0, 2])
        self.assertEqual(len({s.session_id for s in example}), 3)
        self.assertEqual(len(ground_truth), 3)
        print(
            "CASE_STATS sessions",
            len(sessions),
            "users",
            len({s.user_id for s in sessions}),
            "days",
            len(self.clock),
            "active_days",
            len({s.simulation_day for s in sessions}),
            "max_sessions_day",
            max(Counter(s.simulation_day for s in sessions).values()),
        )

    def test_staging_and_isolation(self):
        market = self.market
        asin = market.data.asins[0]
        a = market.open_session("same-user", "first")
        b = market.open_session("same-user", "second")
        self.assertIsNot(a, b)
        before = market.rating(asin)
        self.assertEqual(a.call("rate_product", asin=asin, rating=1)["status"], "ok")
        a.call("publish_review", asin=asin, text="simulated test review")
        a.call("purchase_product", asin=asin)
        self.assertEqual(market.rating(asin), before)
        self.assertFalse(market._visible(asin, "review"))
        self.assertEqual(market.commit_day(self.clock.day(0).end_ms), 3)
        self.assertEqual(len(market._visible(asin, "review")), 1)
        self.assertEqual(len(market._visible(asin, "rating")), 1)
        self.assertEqual(len(market._visible(asin, "purchase")), 1)
        self.assertIn(
            "simulated test review",
            str(
                market.open_session("other", "third").call(
                    "get_reviews", asin=asin, sort="recent", rating_filter="all"
                )
            ),
        )
        fresh = MarketEnvironment(CASE, {"simulation_mode": True})
        try:
            self.assertEqual(fresh.events, [])
            self.assertEqual(fresh.pending_events, [])
        finally:
            fresh.close()
        print("MULTI_SESSION same user: 2 independent; committed 3 writes; review visible next day")

    def test_no_future_series(self):
        market = self.market
        asin = market.data.asins[0]
        future = datetime.fromtimestamp((market.data.t0_ms + DAY_MS) / 1000, timezone.utc)
        with self.assertRaises(ValueError):
            market._value(asin, "csv:NEW", future)
        market.data.series[asin]["csv:NEW"] = [
            {"timestamp_ms": market.data.t0_ms, "value": 1000, "status": "observed"},
            {"timestamp_ms": market.data.t0_ms + DAY_MS, "value": 2000, "status": "observed"},
        ]
        self.assertEqual(market._value(asin, "csv:NEW")["value"], 10)
        market.commit_day(self.clock.day(0).end_ms)
        self.assertEqual(market._value(asin, "csv:NEW")["value"], 20)


if __name__ == "__main__":
    unittest.main()
