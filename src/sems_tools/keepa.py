"""Read only the historical observations in a Keepa product cache.

Keepa timestamps are minutes since 2011-01-01 UTC.  The historical arrays
are alternating [minute, value] pairs, *not* daily records.
"""

from __future__ import annotations
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from typing import Any

EPOCH = datetime(2011, 1, 1, tzinfo=timezone.utc)
SERIES = {
    "new": (1, "USD", 100),
    "used": (2, "USD", 100),
    "list": (4, "USD", 100),
    "rank": (3, "position", 1),
    "rating": (16, "stars", 10),
    "review_count": (17, "count", 1),
}


def keepa_time(minute: int) -> datetime:
    return EPOCH + timedelta(minutes=int(minute))


def cutoff_minutes(day: datetime) -> int:
    return int((day - EPOCH).total_seconds() // 60)


def _decode(raw: Any, key: str) -> float | int | None:
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (float, int)) or raw < 0:
        return None
    scale = SERIES[key][2]
    if key == "rating" and raw == 0:
        return None
    if key == "rank" and raw == 0:
        return None
    if key in ("new", "used", "list") and raw == 0:
        return None
    if scale == 1:
        return int(raw)
    return round(raw / scale, 2)


class KeepaHistory:
    def __init__(self, product: dict):
        self.product = product
        raw_csv = product.get("csv") or []
        self.series: dict[str, tuple[list[int], list[Any]]] = {}
        for key, (index, _, _) in SERIES.items():
            flat = raw_csv[index] if index < len(raw_csv) else None
            if not isinstance(flat, list) or len(flat) % 2:
                self.series[key] = ([], [])
                continue
            # Keepa ordinarily emits sorted pairs. Sort defensively by timestamp;
            # duplicate timestamps retain the last value.
            pairs = sorted(zip(flat[0::2], flat[1::2]), key=lambda x: x[0])
            dedup = {}
            for t, v in pairs:
                if isinstance(t, (int, float)) and not isinstance(t, bool):
                    dedup[int(t)] = v
            keys = sorted(dedup)
            self.series[key] = (keys, [dedup[t] for t in keys])

    def latest(self, key: str, day: datetime) -> dict:
        ts, vals = self.series[key]
        idx = bisect_left(ts, cutoff_minutes(day)) - 1  # never see current day's events
        if idx < 0:
            return {"value": None, "observed_at": None, "state": "not_recorded"}
        val = _decode(vals[idx], key)
        return {
            "value": val,
            "observed_at": keepa_time(ts[idx]).isoformat(),
            "state": "ok" if val is not None else "unavailable",
        }

    def history(self, key: str, day: datetime, window_days: int) -> dict:
        ts, vals = self.series[key]
        start = day - timedelta(days=window_days)
        left = bisect_left(ts, cutoff_minutes(start))
        right = bisect_left(ts, cutoff_minutes(day))
        anchor = self.latest(key, start)
        samples = [
            {
                "observed_at": keepa_time(t).isoformat(),
                "value": _decode(vals[i], key),
            }
            for i, t in enumerate(ts[left:right], start=left)
        ]
        return {
            "window_start": start.date().isoformat(),
            "window_end_exclusive": day.date().isoformat(),
            "value_at_window_start": anchor,
            "observations": samples,
            "latest": self.latest(key, day),
        }

    def rank_context(self) -> dict:
        # The csv[3] primary category can differ from salesRankReference.
        # Derive its category only when exact series content identifies it.
        times, values = self.series["rank"]
        category_id = None
        matches = []
        for cat, flat in (self.product.get("salesRanks") or {}).items():
            if isinstance(flat, list) and flat == [x for pair in zip(times, values) for x in pair]:
                matches.append(str(cat))
        if len(matches) == 1:
            category_id = matches[0]
        cat_names = {
            str(c.get("catId")): c.get("name")
            for c in self.product.get("categoryTree") or []
            if isinstance(c, dict)
        }
        return {
            "category_id": category_id,
            "metadata_sales_rank_reference": self.product.get("salesRankReference"),
            "category_name": cat_names.get(category_id),
            "display_group": self.product.get("salesRankDisplayGroup"),
            "rank_semantics": "Keepa SALES rank, not units sold; ranks across categories are not directly comparable",
        }
