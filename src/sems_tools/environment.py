"""Timestamp-bound shared market and isolated consumer tool sessions."""

from datetime import datetime, timedelta, timezone
from bisect import bisect_right
from math import isfinite
from .case_store import CaseStore
from .registry import ToolRegistry
from .public_trends import PublicTrendHistory

UTC = timezone.utc


def field(value, source="observed", at=None, state=None):
    return {
        "value": value,
        "state": state or ("known" if value is not None else "unknown"),
        "source": source if value is not None else "unknown",
        "observed_at": at,
    }


def parse_time(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamp must have a timezone")
    return value.astimezone(UTC)


class MarketEnvironment:
    def __init__(self, case_path, config=None):
        self.data = CaseStore(case_path)
        config = config or {}
        budget = config.get("query_budget", 6)
        if type(budget) is not int or budget < 0:
            self.data.close()
            raise ValueError("query_budget must be a nonnegative integer")
        self.current_time = self.data.t0
        self.budget = budget
        self.simulation_mode = bool(config.get("simulation_mode", False))
        self.pending_events = []
        self.simulation_finished = False
        self.rank_enabled = bool(config.get("rank_enabled", True))
        self.events = []
        self.sessions = {}
        self.default_asins = self._bsr_order()
        self.trend_history = None
        if config.get("public_trends", False):
            self.enable_public_trends()

    def enable_public_trends(self):
        if self.trend_history is None:
            self.trend_history = PublicTrendHistory(self.current_time)
            self._publish_price_snapshot()

    def _publish_price_snapshot(self):
        if self.trend_history is not None:
            self.trend_history.observe_prices(
                self.current_time, {a: self._value(a, "csv:NEW")["value"] for a in self.data.asins}
            )

    def public_market_context(self):
        if self.trend_history is None:
            raise ValueError("public trend bulletin not enabled")
        return self.trend_history.bulletin(self.current_time, self.events)

    def _valid_bsr(self, asin):
        result = self._value(asin, "csv:SALES", self.data.t0)
        value = result["value"]
        return (
            value
            if result["state"] == "known"
            and result["source"] in ("observed", "interpolated", "filled")
            and type(value) in (int, float)
            and isfinite(value)
            and value > 0
            else None
        )

    def _bsr_order(self):
        # Python's stable sort preserves shelf order for ties and missing values.
        return tuple(
            sorted(
                self.data.asins,
                key=lambda a: (
                    self._valid_bsr(a) is None,
                    self._valid_bsr(a) if self._valid_bsr(a) is not None else 0,
                ),
            )
        )

    def _rank_category(self, asin):
        """Identify the CSV BSR category only when a category series confirms it."""
        value = self._valid_bsr(asin)
        if value is None:
            return None
        matches = []
        for key, series in self.data.series[asin].items():
            if not key.startswith("salesRanks:SALES_RANK_CATEGORY:"):
                continue
            rows = [
                e
                for e in self._read_series(asin, key)
                if e["timestamp_ms"] <= self.data.t0.timestamp() * 1000
            ]
            if (
                rows
                and rows[-1].get("value") == value
                and rows[-1].get("status") not in ("unknown", "unavailable")
            ):
                matches.append(key.rsplit(":", 1)[-1])
        return matches[0] if len(matches) == 1 else None

    def advance_to(self, timestamp, *, _committing=False):
        t = parse_time(timestamp)
        if not self.current_time <= t < self.data.end:
            raise ValueError("time outside forward case window")
        if self.simulation_mode and (int(t.timestamp() * 1000) - self.data.t0_ms) % 86400000:
            raise ValueError("simulation time must be a day boundary")
        if self.simulation_mode and t != self.current_time and not _committing:
            raise ValueError("advance simulation via commit_day")
        self.current_time = t
        self._publish_price_snapshot()
        for s in self.sessions.values():
            s.pages.clear()

    def advance_day(self, day):
        self.advance_to(datetime.fromisoformat(str(day)).replace(tzinfo=UTC))

    def open_session(self, user_id, session_id=None):
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id required")
        key = session_id if session_id is not None else user_id
        if self.simulation_finished:
            raise ValueError("simulation has finished")
        if key not in self.sessions:
            self.sessions[key] = MarketSession(self, user_id, key)
        elif self.sessions[key].user_id != user_id:
            raise ValueError("session ID already belongs to another user")
        return self.sessions[key]

    def commit_day(self, next_day_ms):
        if not self.simulation_mode:
            raise ValueError("day commit requires simulation mode")
        if self.simulation_finished:
            raise ValueError("simulation has finished")
        expected = min(
            self.data.end_ms,
            self.data.t0_ms
            + ((int(self.current_time.timestamp() * 1000) - self.data.t0_ms) // 86400000 + 1)
            * 86400000,
        )
        if next_day_ms != expected:
            raise ValueError("invalid next day boundary")
        visible = datetime.fromtimestamp(next_day_ms / 1000, UTC)
        for event in self.pending_events:
            event["visible_at"] = visible
        count = len(self.pending_events)
        self.events.extend(self.pending_events)
        self.pending_events.clear()
        if next_day_ms < self.data.end_ms:
            self.advance_to(visible, _committing=True)
        else:
            self.simulation_finished = True
        return count

    def _value(self, asin, key, at=None):
        at = at or self.current_time
        if at > self.current_time:
            raise ValueError("future market data requested")
        # Real future reputation/rank outcomes must not overwrite simulated state.
        cutoff = (
            min(at, self.data.t0)
            if not self.simulation_mode or key in ("csv:RATING", "csv:COUNT_REVIEWS", "csv:SALES")
            else at
        ).timestamp() * 1000
        series = self._read_series(asin, key)
        idx = bisect_right([e["timestamp_ms"] for e in series], cutoff) - 1
        if idx < 0:
            return field(None)
        event = series[idx]
        raw = event.get("value")
        source = event.get("status", "unknown")
        if raw is None or source in ("unknown", "unavailable"):
            return field(
                None,
                at=datetime.fromtimestamp(event["timestamp_ms"] / 1000, UTC).isoformat(),
                state=source if source in ("unknown", "unavailable") else "unknown",
            )
        if key in ("csv:NEW", "csv:USED", "csv:LISTPRICE", "csv:AMAZON"):
            raw = round(raw / 100, 2)
        return field(
            raw, source, datetime.fromtimestamp(event["timestamp_ms"] / 1000, UTC).isoformat()
        )

    def _read_series(self, asin, key):
        series = self.data.series[asin].get(key, [])
        if not self.data.case.get("smoothed_values_may_use_future_endpoints"):
            return series
        # A smoothed entry can retain an original -1 alongside an interpolated
        # value. Dropping that entry resurrects an earlier offer. Restore raw
        # observations, including explicit missing states, without interpolation.
        restored = []
        for event in series:
            if "raw_value" not in event:
                if event.get("status") == "observed":
                    restored.append(event)
                continue
            raw = event["raw_value"]
            valid = (
                type(raw) in (int, float)
                and isfinite(raw)
                and raw >= 0
                and (key == "csv:COUNT_REVIEWS" or raw > 0)
            )
            value = (raw / 10 if key == "csv:RATING" else raw) if valid else None
            restored.append(
                {**event, "value": value, "status": "observed" if valid else "unavailable"}
            )
        return restored

    def _visible(self, asin, kind):
        return [
            e
            for e in self.events
            if e["asin"] == asin and e["kind"] == kind and e["visible_at"] <= self.current_time
        ]

    def rating(self, asin):
        avg = self._value(asin, "csv:RATING")
        count = self._value(asin, "csv:COUNT_REVIEWS")
        ratings = self._visible(asin, "rating")
        reviews = self._visible(asin, "review")
        if ratings:
            if avg["value"] is not None and count["value"] is not None and count["value"] > 0:
                avg = field(
                    round(
                        (avg["value"] * count["value"] + sum(x["rating"] for x in ratings))
                        / (count["value"] + len(ratings)),
                        3,
                    ),
                    "filled",
                )
            elif avg["value"] is None:
                avg = field(round(sum(x["rating"] for x in ratings) / len(ratings), 3), "observed")
        if count["value"] is not None:
            count = field(
                count["value"] + len(reviews),
                "filled" if reviews else count["source"],
                count["observed_at"],
            )
        return {"average_rating": avg, "review_count": count}

    def close(self):
        self.data.close()


class MarketSession:
    def __init__(self, market, user_id, session_id=None):
        self.market = market
        self.user_id = user_id
        self.session_id = session_id
        self.remaining = market.budget
        self.order = list(market.default_asins)
        self.initial_order = tuple(self.order)
        self.pages = {}
        self.history = []
        self.sort_history = set()
        self.max_free_sorts = 3
        self.visible_ratings = {}
        self.visible_prices = {a: market._value(a, "csv:NEW") for a in self.order}
        self.registry = ToolRegistry(self)

    def shelf(self):
        self.visible_prices.update({a: self.market._value(a, "csv:NEW") for a in self.order})
        result = {
            "products": [
                {
                    "index": i,
                    "asin": a,
                    "name": next(
                        x["title"] for x in self.market.data.shelf if x["product_id"] == a
                    ),
                    "new_price": self.visible_prices[a],
                }
                for i, a in enumerate(self.order)
            ]
        }
        if self.market.trend_history is not None:
            result["public_market_context"] = self.market.public_market_context()
        return result

    def initial_observation(self):
        return self.shelf()

    def schema(self):
        return self.registry.schema()

    def call(self, name, **args):
        spec = self.registry.schema().get(name)
        error = None
        if not spec:
            error = "unknown or disabled tool"
        else:
            keys = set(spec["params"])
            required = keys - ({"window_days"} if name == "get_price_info" else set())
            if set(args) - keys or required - set(args):
                error = "invalid parameters"
            elif any(
                k in args and (type(args[k]) is not type(v[0]) or args[k] not in v)
                for k, v in spec["params"].items()
                if isinstance(v, list)
            ):
                error = "invalid parameter value"
            elif "asin" in args and args["asin"] not in self.market.data.asins:
                error = "unknown asin"
            elif name == "get_price_info" and (
                (args["view"] == "offers" and "window_days" in args)
                or (args["view"] == "history" and "window_days" not in args)
            ):
                error = "invalid price view combination"
            elif name == "sort_shelf" and args["sort_by"] == "default" and args["order"] != "asc":
                error = "invalid default order"
            elif name == "sort_shelf" and not self.can_sort(args["sort_by"], args["order"]):
                error = "sort unavailable, redundant or free-sort limit reached"
            elif name == "publish_review" and (
                not isinstance(args["text"], str) or not args["text"].strip()
            ):
                error = "text required"
            elif name == "rate_product" and type(args["rating"]) is not int:
                error = "invalid rating"
            elif spec["kind"] == "read" and self.remaining <= 0:
                error = "budget exhausted"
        if error:
            return {
                "tool": name,
                "status": "invalid_request",
                "error": error,
                "data": None,
                "cost": 0,
            }
        cost = 1 if spec["kind"] == "read" else 0
        if cost:
            self.remaining -= cost
        data = getattr(self, "_" + name)(**args)
        if name == "get_rating_info":
            self.visible_ratings[args["asin"]] = data
        if name == "get_price_info":
            current = (
                (data.get("offers") or {}).get("new")
                if args["view"] == "offers"
                else data.get("current")
            )
            if current is not None:
                self.visible_prices[args["asin"]] = current
        if name == "sort_shelf":
            self.sort_history.add((args["sort_by"], args["order"]))
        result = {
            "tool": name,
            "status": "ok",
            "data": data,
            "cost": cost,
            "remaining": self.remaining,
            "as_of": self.market.current_time.isoformat(),
        }
        self.history.append({"tool": name, "args": args, "result": result})
        return result

    def _get_product_details(self, asin, section):
        p = self.market.data.products[asin]
        static = p.get("keepa", {}).get("static", {})
        keys = (
            ("description", "features")
            if section == "overview"
            else ("brand", "model", "size", "color", "manufacturer")
        )
        return {
            **{
                k: field(static.get(k), state="unavailable" if k not in static else None)
                for k in keys
            },
            "historical_snapshot_verified": False,
        }

    def _get_price_info(self, asin, view, window_days=None):
        if view == "offers":
            return {
                "currency": "USD",
                "offers": {
                    k: self.market._value(asin, "csv:" + v)
                    for k, v in [("new", "NEW"), ("used", "USED"), ("list", "LISTPRICE")]
                },
            }
        end = (
            self.market.current_time
            if self.market.simulation_mode
            else min(self.market.current_time, self.market.data.t0)
        )
        start = end - timedelta(days=window_days)
        rows = []
        for e in self.market._read_series(asin, "csv:NEW"):
            t = datetime.fromtimestamp(e["timestamp_ms"] / 1000, UTC)
            if start <= t <= end:
                rows.append(
                    {
                        "at": t.isoformat(),
                        "price": field(
                            round(e["value"] / 100, 2) if e.get("value") is not None else None,
                            e.get("status", "unknown"),
                            t.isoformat(),
                            state="unavailable" if e.get("status") == "unavailable" else None,
                        ),
                    }
                )
        return {
            "window_days": window_days,
            "observations": rows,
            "current": self.market._value(asin, "csv:NEW"),
        }

    def _get_rating_info(self, asin):
        return self.market.rating(asin)

    def _get_reviews(self, asin, sort, rating_filter):
        key = (asin, sort, rating_filter)
        offset = self.pages.get(key, 0)
        raw = []
        for r in self.market.data.reviews[asin]:
            if (
                r.get("timestamp_ms", float("inf")) < self.market.data.t0.timestamp() * 1000
                and r.get("timestamp_ms", float("inf"))
                <= self.market.current_time.timestamp() * 1000
            ):
                raw.append(
                    {
                        "id": r.get("review_id"),
                        "text": r.get("text"),
                        "rating": r.get("rating"),
                        "at": r.get("review_timestamp_utc"),
                        "ms": r["timestamp_ms"],
                    }
                )
        for e in self.market._visible(asin, "review"):
            rating = next(
                (
                    x["rating"]
                    for x in self.market._visible(asin, "rating")
                    if x.get("session_id") == e.get("session_id") and x["user_id"] == e["user_id"]
                ),
                None,
            )
            raw.append(
                {
                    "id": e["id"],
                    "text": e["text"],
                    "rating": rating,
                    "at": e["at"],
                    "ms": datetime.fromisoformat(e["at"]).timestamp() * 1000,
                }
            )

        def matches(r):
            v = r["rating"]
            return rating_filter == "all" or (
                v is not None
                and (
                    v >= 4
                    if rating_filter == "positive"
                    else v <= 2
                    if rating_filter == "negative"
                    else v == int(rating_filter)
                )
            )

        raw = [r for r in raw if matches(r)]
        raw.sort(key=lambda r: (r["ms"], str(r["id"])), reverse=sort == "recent")
        batch = raw[offset : offset + 5] if offset >= 0 else []
        self.pages[key] = offset + 5 if offset + 5 < len(raw) else -1
        return {
            "reviews": [{k: v for k, v in r.items() if k != "ms"} for r in batch],
            "has_more": self.pages[key] != -1,
            "state": "known" if raw else "no_matches",
        }

    def _get_rank_info(self, asin):
        market = self.market
        rank = market._value(asin, "csv:SALES")
        valid = market._valid_bsr(asin)
        if valid is None and rank["value"] is not None:
            rank = field(None, state="unavailable")
        valid_order = [a for a in market.default_asins if market._valid_bsr(a) is not None]
        bsr_order = field(valid_order.index(asin) + 1) if asin in valid_order else field(None)
        category = market._rank_category(asin)
        comparable = (
            [a for a in valid_order if market._rank_category(a) == category] if category else []
        )
        relative = (
            field(comparable.index(asin) + 1)
            if category and len(comparable) > 1
            else field(None, state="not_applicable")
        )
        purchases = {a: len(market._visible(a, "purchase")) for a in market.data.asins}
        simulated_order = sorted(
            market.data.asins, key=lambda a: (-purchases[a], market.data.asins.index(a))
        )
        return {
            "amazon_bsr": rank,
            "case_bsr_order": bsr_order,
            "amazon_bsr_scope": "pre_t0_baseline; not simulated live sales"
            if market.simulation_mode
            else "historical_baseline",
            "case_relative_rank": relative,
            "bsr_category": field(category, state="known" if category else "unknown"),
            "simulated_purchase_count": purchases[asin],
            "simulated_case_rank": field(simulated_order.index(asin) + 1),
        }

    def _sorted_order(self, sort_by, order):
        if sort_by == "default":
            return list(self.initial_order)

        def value(asin):
            field_value = (
                self.visible_prices[asin]
                if sort_by == "price"
                else self.visible_ratings.get(asin, {}).get(
                    "average_rating" if sort_by == "rating" else "review_count", {}
                )
            )
            return field_value.get("value") if field_value.get("state") == "known" else None

        return sorted(
            self.order,
            key=lambda a: (
                value(a) is None,
                -value(a)
                if value(a) is not None and order == "desc"
                else value(a)
                if value(a) is not None
                else 0,
            ),
        )

    def can_sort(self, sort_by, order):
        if len(self.sort_history) >= self.max_free_sorts or (sort_by, order) in self.sort_history:
            return False
        if (
            sort_by not in ("default", "price", "rating", "rating_count")
            or order not in ("asc", "desc")
            or (sort_by == "default" and order != "asc")
        ):
            return False
        if sort_by in ("rating", "rating_count"):
            key = "average_rating" if sort_by == "rating" else "review_count"
            if not any(
                self.visible_ratings.get(a, {}).get(key, {}).get("state") == "known"
                for a in self.order
            ):
                return False
        return self._sorted_order(sort_by, order) != self.order

    def _sort_shelf(self, sort_by, order):
        self.order = self._sorted_order(sort_by, order)
        return self.shelf()

    def _write(self, kind, asin, **extra):
        now = self.market.current_time
        if self.market.simulation_mode:
            start = self.market.data.t0_ms
            now_ms = int(now.timestamp() * 1000)
            visible = datetime.fromtimestamp(
                min(self.market.data.end_ms, start + ((now_ms - start) // 86400000 + 1) * 86400000)
                / 1000,
                UTC,
            )
        else:
            tomorrow = (now + timedelta(days=1)).date()
            visible = datetime.combine(tomorrow, datetime.min.time(), UTC)
        event = {
            "kind": kind,
            "asin": asin,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "at": now.isoformat(),
            "visible_at": visible,
            **extra,
        }
        (self.market.pending_events if self.market.simulation_mode else self.market.events).append(
            event
        )
        return {"accepted": True, "visible_at": visible.isoformat()}

    def _purchase_product(self, asin):
        return self._write("purchase", asin)

    def _rate_product(self, asin, rating):
        return self._write("rating", asin, rating=rating)

    def _publish_review(self, asin, text):
        return self._write(
            "review",
            asin,
            text=text,
            id="sim_" + str(len(self.market.events) + len(self.market.pending_events) + 1),
        )
