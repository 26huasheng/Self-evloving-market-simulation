"""Explicit public case-market bulletin, built only from published observations.

No individual-product histories, review bodies, hidden purchases, arrivals,
ground-truth reviews or interpolation endpoints are published by this channel.
"""

from datetime import timedelta
from statistics import median, pstdev
import math

TREND_SCHEMA = "public_market_trends_v1"
TREND_NAMES = (
    "review_activity_change",
    "price_change",
    "price_volatility_change",
    "price_coverage_change",
)
TREND_DEFINITIONS = {
    "review_activity_change": "(recent-previous)/(recent+previous+1) committed simulated reviews in two adjacent windows; positive means increasing activity, not sales.",
    "price_change": "tanh of median log(current/previous) published price for matched products one window apart; positive means rising prices.",
    "price_volatility_change": "tanh of recent minus previous standard deviation of daily median matched-product log returns; positive means more volatile.",
    "price_coverage_change": "Current minus previous fraction of shelf products with valid public prices; positive means improving data coverage, not demand.",
}


def measure(value=None, *, source, support=0):
    return {"value": value, "known": value is not None, "source": source, "support": support}


class PublicTrendHistory:
    def __init__(self, start, window_days=7):
        if type(window_days) is not int or window_days < 1:
            raise ValueError("trend window must be a positive integer")
        self.start = start
        self.window_days = window_days
        self.snapshots = {}

    def observe_prices(self, at, prices):
        # A day's published snapshot is immutable: queries cannot rewrite it.
        self.snapshots.setdefault(
            at,
            {
                key: value
                if type(value) in (int, float) and math.isfinite(value) and value > 0
                else None
                for key, value in prices.items()
            },
        )

    def bulletin(self, at, events):
        window = timedelta(days=self.window_days)
        current = self.snapshots[at]
        previous = self.snapshots.get(at - window)
        values = {name: measure(source="published_default_shelf_prices") for name in TREND_NAMES}
        values["review_activity_change"] = measure(source="committed_simulated_reviews")
        if previous is not None:
            paired = [
                (previous[a], current[a])
                for a in sorted(current.keys() & previous.keys())
                if previous[a] is not None
                and current[a] is not None
                and previous[a] > 0
                and current[a] > 0
            ]
            if paired:
                # Bounded signed log return. Identical identities on both dates
                # prevent product-entry/missingness changes mimicking a price move.
                values["price_change"] = measure(
                    math.tanh(median(math.log(b / a) for a, b in paired)),
                    source="published_default_shelf_prices",
                    support=len(paired),
                )
            values["price_coverage_change"] = measure(
                (
                    sum(v is not None for v in current.values())
                    - sum(v is not None for v in previous.values())
                )
                / max(len(current), 1),
                source="published_default_shelf_prices",
                support=len(current),
            )
        if at - self.start >= 2 * window:
            # Use publication time, including events that became public today.
            published = [
                e for e in events if e["kind"] == "review" and self.start < e["visible_at"] <= at
            ]
            recent = sum(at - window < e["visible_at"] <= at for e in published)
            before = sum(at - 2 * window < e["visible_at"] <= at - window for e in published)
            values["review_activity_change"] = measure(
                (recent - before) / (recent + before + 1),
                source="committed_simulated_reviews",
                support=recent + before,
            )
            returns = []
            for index in range(2 * self.window_days):
                right = at - timedelta(days=2 * self.window_days - 1 - index)
                b = self.snapshots.get(right)
                a = self.snapshots.get(right - timedelta(days=1))
                pairs = (
                    [
                        (a[k], b[k])
                        for k in sorted(a.keys() & b.keys())
                        if a[k] is not None and b[k] is not None and a[k] > 0 and b[k] > 0
                    ]
                    if a is not None and b is not None
                    else []
                )
                returns.append(median(math.log(y / x) for x, y in pairs) if pairs else None)
            if all(x is not None for x in returns):
                values["price_volatility_change"] = measure(
                    math.tanh(
                        pstdev(returns[self.window_days :]) - pstdev(returns[: self.window_days])
                    ),
                    source="published_default_shelf_prices",
                    support=2 * self.window_days,
                )
        return {
            "schema": TREND_SCHEMA,
            "as_of": at.isoformat(),
            "window_days": self.window_days,
            "scope": "public aggregate case-market changes; simulated review activity is not actual sales",
            "features": values,
            "feature_definitions": dict(TREND_DEFINITIONS),
            "warmup_note": f"Price changes require a {self.window_days}-day baseline; review/volatility changes require {2 * self.window_days} days. Unknown is not zero change.",
        }
