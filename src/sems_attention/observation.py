"""Consumer-visible cumulative state. No reads from market ground state."""

from dataclasses import dataclass, field
from collections import defaultdict
import math
import numpy as np

FEATURE_SCHEMA = "attention_features_v1"
# price log, price known, shelf position, rating/5, rating status,
# review count log, review count status, BSR log, BSR status,
# read star mean, read star present, review pages, detail obtained,
# price obtained, rating obtained, rank obtained.
PRODUCT_FEATURES = (
    "price_log",
    "price_known",
    "position",
    "rating",
    "rating_known",
    "count_log",
    "count_known",
    "bsr_log",
    "bsr_known",
    "read_star_mean",
    "read_star_known",
    "review_pages",
    "detail_seen",
    "price_seen",
    "rating_seen",
    "rank_seen",
)
STATUS = {"not_observed": 0, "known": 1, "unknown": 2, "unavailable": 3, "not_applicable": 4}
SOURCE = {"unknown": 0, "observed": 1, "interpolated": 2, "filled": 3}


def status_code(field):
    if field is None:
        return 0.0  # not_observed
    return (
        STATUS.get(field.get("state", "unknown"), 2)
        + SOURCE.get(field.get("source", "unknown"), 0) / 10
    ) / 5


def numeric(field):
    return (
        field.get("value")
        if isinstance(field, dict)
        and field.get("state") == "known"
        and isinstance(field.get("value"), (float, int))
        else None
    )


@dataclass
class Observation:
    initial_shelf: dict
    remaining: int
    initial_budget: int
    market_time: str
    profile: dict
    order: list = field(default_factory=list)
    results: list = field(default_factory=list)
    details: dict = field(default_factory=dict)
    prices: dict = field(default_factory=dict)
    ratings: dict = field(default_factory=dict)
    ranks: dict = field(default_factory=dict)
    reviews: dict = field(default_factory=lambda: defaultdict(dict))
    calls: dict = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self):
        self.order = [p["asin"] for p in self.initial_shelf["products"]]
        self.cards = {p["asin"]: dict(p) for p in self.initial_shelf["products"]}

    def update(self, action, result):
        self.results.append({"action": action.as_dict(), "result": result})
        if result["status"] != "ok":
            return
        self.remaining = result["remaining"]
        name = action.tool_name
        asin = action.asin
        data = result["data"]
        self.calls[name] += 1
        if name == "sort_shelf":
            self.order = [p["asin"] for p in data["products"]]
            self.cards.update({p["asin"]: dict(p) for p in data["products"]})
        elif name == "get_product_details":
            self.details.setdefault(asin, {})[action.parameters["section"]] = data
        elif name == "get_price_info":
            self.prices.setdefault(asin, {})[
                (action.parameters["view"], action.parameters.get("window_days"))
            ] = data
            current = (
                (data.get("offers") or {}).get("new")
                if action.parameters["view"] == "offers"
                else data.get("current")
            )
            if current is not None:
                self.cards[asin]["new_price"] = current
        elif name == "get_rating_info":
            self.ratings[asin] = data
        elif name == "get_rank_info":
            self.ranks[asin] = data
        elif name == "get_reviews":
            for row in data["reviews"]:
                self.reviews[asin][row["id"]] = row

    def product_features(self, asin):
        card = self.cards[asin]
        price = numeric(card["new_price"])
        rating = self.ratings.get(asin, {})
        rank = self.ranks.get(asin, {})
        stars = [
            r["rating"]
            for r in self.reviews.get(asin, {}).values()
            if isinstance(r.get("rating"), (int, float))
        ]
        r = numeric(rating.get("average_rating"))
        c = numeric(rating.get("review_count"))
        b = numeric(rank.get("amazon_bsr"))
        count = sum(
            x["action"]["tool_name"] == "get_reviews" and x["action"]["asin"] == asin
            for x in self.results
        )
        vals = [
            math.log1p(price) / 8 if price is not None else 0,
            status_code(card["new_price"]),
            (self.order.index(asin) + 1) / max(len(self.order), 1),
            r / 5 if r is not None else 0,
            status_code(rating.get("average_rating")),
            math.log1p(c) / 12 if c is not None else 0,
            status_code(rating.get("review_count")),
            math.log1p(b) / 20 if b is not None else 0,
            status_code(rank.get("amazon_bsr")),
            sum(stars) / len(stars) / 5 if stars else 0,
            float(bool(stars)),
            min(count / 10, 1),
            float(asin in self.details),
            float(asin in self.prices),
            float(asin in self.ratings),
            float(asin in self.ranks),
        ]
        return np.asarray(vals, dtype="float32")

    def product_texts(self, asin):
        card = self.cards[asin]
        details = self.details.get(asin, {})
        detail = " ".join(
            str(f["value"])
            for section in details.values()
            for f in section.values()
            if isinstance(f, dict) and f.get("value") is not None
        )
        return (
            card["name"],
            detail,
            [str(r.get("text") or "") for r in self.reviews.get(asin, {}).values()],
        )

    def shelf_stats(self):
        prices = [numeric(self.cards[a]["new_price"]) for a in self.order]
        valid = [x for x in prices if x is not None]
        n = max(len(self.order), 1)
        return np.asarray(
            [
                min(math.log1p(n) / 8, 1),
                sum(x is not None for x in prices) / n,
                math.log1p(np.mean(valid)) / 8 if valid else 0,
                math.log1p(max(valid)) / 8 if valid else 0,
                len(self.details) / n,
                len(self.ratings) / n,
                sum(bool(v) for v in self.reviews.values()) / n,
                self.remaining / max(self.initial_budget, 1),
            ],
            dtype="float32",
        )

    def query_state(self):
        names = (
            "get_product_details",
            "get_price_info",
            "get_rating_info",
            "get_reviews",
            "get_rank_info",
            "sort_shelf",
        )
        values = [
            self.remaining / max(self.initial_budget, 1),
            min(len(self.results) / max(self.initial_budget, 1), 1),
        ] + [min(self.calls[n] / max(self.initial_budget, 1), 1) for n in names]
        values += [
            float(bool(self.details)),
            float(bool(self.prices)),
            float(bool(self.ratings)),
            float(bool(self.ranks)),
            float(any(self.reviews.values())),
            float(self.remaining == 0),
            min(len(self.results) / 20, 1),
            1.0,
        ]
        return np.asarray(values, dtype="float32")

    def decision_observation(self):
        result = {
            "profile": self.profile,
            "market_time": self.market_time,
            "initial_shelf": self.initial_shelf,
            "shelf": [
                {
                    "product_id": a,
                    "title": self.cards[a]["name"],
                    "price": self.cards[a]["new_price"],
                }
                for a in self.order
            ],
            "tool_results": self.results,
            "read_reviews": {a: list(rows.values()) for a, rows in self.reviews.items()},
            "remaining_budget": self.remaining,
        }
        if "public_market_context" in self.initial_shelf:
            result["public_market_context"] = self.initial_shelf["public_market_context"]
        return result

    def for_decision_model(self):
        obs = self.decision_observation()
        result = {
            "shelf": obs["shelf"],
            "tool_data": {
                r["action"]["tool_name"]: r["result"]["data"]
                for r in self.results
                if r["result"]["status"] == "ok"
            },
        }
        if "public_market_context" in obs:
            result["public_market_context"] = obs["public_market_context"]
        return result

    def decision_inputs(self):
        """Arguments for existing DecisionModel.decide(agent_id, profile, observation)."""

        class FixedProfile:
            def __init__(self, data):
                self.data = data

            def to_dict(self):
                return self.data

        return self.profile["user_id"], FixedProfile(self.profile), self.for_decision_model()
