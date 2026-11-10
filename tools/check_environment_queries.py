"""Check local environment queries only; no Attention, Decision, training or API."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

from sems_tools import MarketEnvironment

DAY = 86400000


def has_content(tool, data, args):
    if tool == "get_reviews":
        return bool(data["reviews"])
    if tool == "get_product_details":
        return any(
            isinstance(v, dict) and v.get("value") not in (None, "", []) for v in data.values()
        )
    if tool == "get_price_info":
        fields = (
            list(data["offers"].values())
            if args["view"] == "offers"
            else [r["price"] for r in data["observations"]]
        )
        return any(v.get("state") == "known" and v.get("value") is not None for v in fields)
    if tool == "get_rank_info":
        return data["amazon_bsr"]["value"] is not None
    return any(
        isinstance(v, dict) and v.get("state") == "known" and v.get("value") is not None
        for v in data.values()
    )


def check_payload(market, tool, args, data, offset=0):
    asin = args.get("asin")
    if tool == "get_product_details":
        static = market.data.products[asin].get("keepa", {}).get("static", {})
        for key, value in data.items():
            if key != "historical_snapshot_verified":
                assert value["value"] == static.get(key), "details disagree with source"
    elif tool == "get_reviews":
        rows = [
            r
            for r in market.data.reviews[asin]
            if r.get("timestamp_ms", math.inf) < market.data.t0_ms
        ]
        filt = args["rating_filter"]
        if filt != "all":
            rows = [
                r
                for r in rows
                if r.get("rating") is not None
                and (
                    r["rating"] >= 4
                    if filt == "positive"
                    else r["rating"] <= 2
                    if filt == "negative"
                    else r["rating"] == int(filt)
                )
            ]
        rows.sort(
            key=lambda r: (r["timestamp_ms"], str(r.get("review_id"))),
            reverse=args["sort"] == "recent",
        )
        expected = rows[offset : offset + 5]
        assert [(r["id"], r["text"], r["rating"]) for r in data["reviews"]] == [
            (r.get("review_id"), r.get("text", ""), r.get("rating")) for r in expected
        ], "reviews disagree with source/filter/order"
        assert data["has_more"] == (offset + 5 < len(rows))
    elif tool == "get_price_info":
        if args["view"] == "history":
            end = market.current_time
            start_ms = int(end.timestamp() * 1000) - args["window_days"] * DAY
            assert all(
                start_ms
                <= int(datetime.fromisoformat(r["at"]).timestamp() * 1000)
                <= int(end.timestamp() * 1000)
                for r in data["observations"]
            )
        fields = (
            data["offers"].values()
            if args["view"] == "offers"
            else [data["current"], *[r["price"] for r in data["observations"]]]
        )
        for value in fields:
            if value["value"] is not None:
                assert (
                    isinstance(value["value"], (int, float))
                    and math.isfinite(value["value"])
                    and value["value"] > 0
                )
            if value.get("observed_at"):
                assert datetime.fromisoformat(value["observed_at"]) <= market.current_time
    elif tool == "get_rating_info":
        average = data["average_rating"]["value"]
        assert average is None or 1 <= average <= 5
    elif tool == "get_rank_info":
        rank = data["amazon_bsr"]["value"]
        assert rank is None or rank > 0
        assert data["simulated_purchase_count"] == 0


def check_case(path, trace):
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    market = MarketEnvironment(path, {"simulation_mode": True, "query_budget": 100000})
    tools = {}
    errors = []
    calls = 0
    try:
        days = (market.data.end_ms - market.data.t0_ms + DAY - 1) // DAY
        targets = sorted({0, days // 2, days - 1})
        for day in range(targets[-1] + 1):
            if day in targets:
                session = market.open_session("query_check", f"day_{day}")
                shelf = session.initial_observation()
                assert len(shelf["products"]) == len(market.data.asins)
                actions = [
                    a
                    for a in session.registry.actions()
                    if a["allowed"] and a["tool"] not in ("STOP", "sort_shelf")
                ]
                for action in actions:
                    name, args = action["tool"], action["args"]
                    # Also verify a second page when present, with exact source comparison.
                    offset = 0
                    while True:
                        before = session.remaining
                        response = session.registry.call(name, **args)
                        calls += 1
                        key = name + (
                            ":" + str(args.get("section", args.get("view", "")))
                            if name in ("get_product_details", "get_price_info")
                            else ""
                        )
                        stats = tools.setdefault(key, Counter())
                        stats["calls"] += 1
                        try:
                            assert response["status"] == "ok", response
                            assert response["cost"] == 1 and session.remaining == before - 1
                            assert response["as_of"] == market.current_time.isoformat()
                            check_payload(market, name, args, response["data"], offset)
                            stats["ok"] += 1
                            stats[
                                "with_content"
                                if has_content(name, response["data"], args)
                                else "empty_or_unknown"
                            ] += 1
                        except Exception as exc:
                            errors.append(
                                {"day": day, "tool": name, "args": args, "error": str(exc)}
                            )
                            stats["failed"] += 1
                        trace.write(
                            json.dumps(
                                {
                                    "case_id": market.data.case_id,
                                    "day": day,
                                    "tool": name,
                                    "args": args,
                                    "page_offset": offset,
                                    "response": response,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        if (
                            name == "get_reviews"
                            and args["sort"] == "recent"
                            and args["rating_filter"] == "all"
                            and offset == 0
                            and (response.get("data") or {}).get("has_more")
                        ):
                            offset = 5
                            continue
                        break
                # Execute only currently available free sorts; rating queries above unlock rating-based sorts.
                for _ in range(session.max_free_sorts):
                    available = [
                        a
                        for a in session.registry.actions()
                        if a["tool"] == "sort_shelf" and a["allowed"]
                    ]
                    if not available:
                        break
                    action = available[0]
                    before = session.remaining
                    response = session.registry.call("sort_shelf", **action["args"])
                    calls += 1
                    assert (
                        response["status"] == "ok"
                        and response["cost"] == 0
                        and session.remaining == before
                    )
                    assert set(p["asin"] for p in response["data"]["products"]) == set(
                        market.data.asins
                    )
                    tools.setdefault("sort_shelf", Counter()).update(calls=1, ok=1, with_content=1)
                    trace.write(
                        json.dumps(
                            {
                                "case_id": market.data.case_id,
                                "day": day,
                                "tool": "sort_shelf",
                                "args": action["args"],
                                "response": response,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if day < targets[-1]:
                market.commit_day(min(market.data.end_ms, market.data.t0_ms + (day + 1) * DAY))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
        return {
            "path": str(path),
            "case_id": market.data.case_id,
            "products": len(market.data.asins),
            "checked_days": targets,
            "calls": calls,
            "tools": tools,
            "errors": errors,
            "source_unchanged": True,
        }
    finally:
        market.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = args.case
    if not cases or any(not p.is_file() for p in cases):
        parser.error("all input Cases must exist")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    with (
        patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
        (args.output_dir / "queries.jsonl").open("w", encoding="utf-8") as trace,
    ):
        for path in cases:
            result = check_case(path, trace)
            results.append(result)
            print(
                "CASE",
                path.stem,
                result["calls"],
                "calls",
                len(result["errors"]),
                "errors",
                flush=True,
            )
    report = {
        "scope": "local environment and queries only",
        "network_blocked": True,
        "api_calls": 0,
        "cases": results,
        "calls": sum(r["calls"] for r in results),
        "errors": sum(len(r["errors"]) for r in results),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "SUMMARY",
        {
            "cases": len(results),
            "calls": report["calls"],
            "errors": report["errors"],
            "output": str(args.output_dir),
        },
    )
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
