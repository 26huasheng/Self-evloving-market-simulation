from datetime import timedelta
from pathlib import Path
import hashlib
import pytest
from sems_tools import MarketEnvironment

CASE = Path(__import__("os").environ["SEMS_CASE_PATH"])


@pytest.fixture
def market():
    m = MarketEnvironment(CASE, {"query_budget": 100})
    yield m
    m.close()


def test_loader_shelf_and_mapping(market):
    assert len(market.data.asins) == len(market.data.shelf)
    assert market.data.users
    s = market.open_session("one")
    assert s.remaining == 100
    assert [x["asin"] for x in s.shelf()["products"]] == list(market.default_asins)
    assert all(s.registry.asin(i) == a for i, a in enumerate(market.data.asins))
    assert all("focal" not in str(x).lower() for x in s.shelf()["products"])


def test_all_read_combinations(market):
    s = market.open_session("one")
    a = market.data.asins[0]
    for section in ("overview", "metadata"):
        assert s.call("get_product_details", asin=a, section=section)["status"] == "ok"
    assert s.call("get_price_info", asin=a, view="offers")["status"] == "ok"
    for days in (30, 90):
        x = s.call("get_price_info", asin=a, view="history", window_days=days)
        assert x["status"] == "ok"
        assert all(
            row["at"] <= market.current_time.isoformat() for row in x["data"]["observations"]
        )
    assert s.call("get_rating_info", asin=a)["status"] == "ok"
    assert s.call("get_rank_info", asin=a)["status"] == "ok"
    for sort in ("recent", "oldest"):
        for filt in ("all", "positive", "negative", "1", "2", "3", "4", "5"):
            x = s.call("get_reviews", asin=a, sort=sort, rating_filter=filt)
            assert x["status"] == "ok"
            assert len(x["data"]["reviews"]) <= 5
            for r in x["data"]["reviews"]:
                if filt == "positive":
                    assert r["rating"] >= 4
                if filt == "negative":
                    assert r["rating"] <= 2
                if filt.isdigit():
                    assert r["rating"] == int(filt)


def test_paging_budget_and_invalid(market):
    a = market.data.asins[0]
    s = market.open_session("one")
    x = s.call("get_reviews", asin=a, sort="oldest", rating_filter="all")
    y = s.call("get_reviews", asin=a, sort="oldest", rating_filter="all")
    assert x["data"]["reviews"] != y["data"]["reviews"]
    assert (
        s.call("get_price_info", asin=a, view="history", window_days=15)["status"]
        == "invalid_request"
    )
    assert (
        s.call("get_reviews", asin=a, sort="recent", rating_filter="all", cursor="x")["status"]
        == "invalid_request"
    )
    assert s.call("get_rating_info", asin=a, as_of="2020-01-01")["status"] == "invalid_request"
    assert s.remaining == 98
    m = MarketEnvironment(CASE, {"query_budget": 1})
    q = m.open_session("q")
    assert q.call("get_rating_info", asin=a)["status"] == "ok"
    assert q.call("get_rating_info", asin=a)["status"] == "invalid_request"
    assert any(x["tool"] == "STOP" and x["allowed"] for x in q.registry.actions())
    m.close()


def test_writes_visibility_isolation_and_source_integrity(market):
    before = hashlib.sha256(CASE.read_bytes()).digest()
    a = market.data.asins[0]
    s = market.open_session("one")
    other = market.open_session("two")
    old = other.call("get_rating_info", asin=a)["data"]
    assert s.call("purchase_product", asin=a)["status"] == "ok"
    assert s.call("rate_product", asin=a, rating=5)["status"] == "ok"
    assert s.call("publish_review", asin=a, text="A test review")["status"] == "ok"
    assert other.call("get_rating_info", asin=a)["data"] == old
    assert other.call("get_rank_info", asin=a)["data"]["simulated_purchase_count"] == 0
    market.advance_to(market.current_time + timedelta(days=1))
    assert other.call("get_rank_info", asin=a)["data"]["simulated_purchase_count"] == 1
    assert (
        other.call("get_reviews", asin=a, sort="recent", rating_filter="5")["data"]["reviews"][0][
            "text"
        ]
        == "A test review"
    )
    assert hashlib.sha256(CASE.read_bytes()).digest() == before
    assert s.order is not other.order


def test_rank_switch_and_sort(market):
    a = market.data.asins[0]
    s = market.open_session("one")
    other = market.open_session("two")
    assert s.call("sort_shelf", sort_by="rating", order="desc")["status"] == "invalid_request"
    sorts = [x for x in s.registry.actions() if x["tool"] == "sort_shelf" and x["allowed"]]
    assert sorts
    args = sorts[0]["args"]
    result = s.call("sort_shelf", **args)
    assert result["status"] == "ok" and result["cost"] == 0 and s.remaining == market.budget
    assert s.call("sort_shelf", **args)["status"] == "invalid_request"
    assert other.order == list(market.default_asins)
    m = MarketEnvironment(CASE, {"rank_enabled": False})
    q = m.open_session("q")
    assert "get_rank_info" not in q.schema()
    assert q.call("get_rank_info", asin=a)["status"] == "invalid_request"
    m.close()


def test_unknown_fields_and_exact_timestamp(market):
    from datetime import datetime, timezone

    a = market.data.asins[0]
    series = market.data.series[a]["csv:NEW"]
    observed = [
        e
        for e in series
        if e.get("value") is not None and e["timestamp_ms"] <= market.data.t0.timestamp() * 1000
    ]
    assert observed
    point = observed[-1]["timestamp_ms"] / 1000
    before = market._value(a, "csv:NEW", datetime.fromtimestamp(point - 1, timezone.utc))
    after = market._value(a, "csv:NEW", datetime.fromtimestamp(point, timezone.utc))
    assert after["observed_at"] == datetime.fromtimestamp(point, timezone.utc).isoformat()
    assert before["observed_at"] != after["observed_at"]
    original = market.data.series[a]
    market.data.series[a] = {**original, "csv:RATING": []}
    try:
        rating = market.open_session("unknown").call("get_rating_info", asin=a)["data"]
        assert rating["average_rating"]["state"] == "unknown"
        assert rating["review_count"]["state"] in ("known", "unknown")
        assert rating["average_rating"]["value"] is None
    finally:
        market.data.series[a] = original


def test_bsr_order_missing_ties_and_simulated_rank(market):
    asins = market.data.asins
    originals = {a: market.data.series[a] for a in asins[:4]}
    timestamp = int(market.data.t0.timestamp() * 1000)
    values = (50, 50, None, None)
    try:
        for asin, value in zip(asins[:4], values):
            market.data.series[asin] = {
                **originals[asin],
                "csv:SALES": [
                    {
                        "timestamp_ms": timestamp,
                        "value": value,
                        "status": "observed" if value is not None else "unknown",
                    }
                ],
            }
        market.default_asins = market._bsr_order()
        valid = [a for a in market.default_asins if market._valid_bsr(a) is not None]
        assert [market._valid_bsr(a) for a in valid] == sorted(market._valid_bsr(a) for a in valid)
        missing_order = [a for a in market.default_asins if market._valid_bsr(a) is None]
        assert missing_order == [a for a in asins if market._valid_bsr(a) is None]
        assert market.default_asins.index(asins[0]) < market.default_asins.index(asins[1])
        s = market.open_session("bsr_test")
        for index, asin in enumerate(valid, 1):
            assert s.call("get_rank_info", asin=asin)["data"]["case_bsr_order"]["value"] == index
        missing = s.call("get_rank_info", asin=asins[2])["data"]
        assert missing["amazon_bsr"]["value"] is None
        assert missing["case_bsr_order"]["value"] is None
        assert missing["case_relative_rank"]["state"] == "not_applicable"
        assert s.call("purchase_product", asin=asins[2])["status"] == "ok"
        market.advance_to(market.current_time + timedelta(days=1))
        result = s.call("get_rank_info", asin=asins[2])["data"]
        assert result["simulated_purchase_count"] == 1
        assert result["simulated_case_rank"]["value"] == 1
        assert result["case_bsr_order"]["value"] is None
    finally:
        for asin, series in originals.items():
            market.data.series[asin] = series
