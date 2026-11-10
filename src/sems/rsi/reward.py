"""Observable-information pilot reward; ground truth is used only after Decision."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RewardConfig:
    scheme: str = "B"
    rating_weight: float = 0.05
    alpha: float = 0.01
    beta: float = 0.005
    max_information: int = 10
    max_reviews_per_page: int = 3

    def __post_init__(self):
        import math

        if (
            self.scheme not in ("A", "B", "C")
            or not all(
                math.isfinite(x) and x >= 0 for x in (self.alpha, self.beta, self.rating_weight)
            )
            or type(self.max_information) is not int
            or self.max_information < 0
            or type(self.max_reviews_per_page) is not int
            or self.max_reviews_per_page < 1
        ):
            raise ValueError("invalid reward config")


def _known(value):
    return (
        isinstance(value, dict) and value.get("state") == "known" and value.get("value") is not None
    )


def information_units(step):
    """Keys for first-observed facts. Never reads labels, text truth or hidden data."""
    if step.get("tool") == "STOP" or step.get("result_status") != "ok":
        return set()
    tool = step.get("tool")
    asin = step.get("asin")
    data = (step.get("result") or {}).get("data") or {}
    units = set()
    if tool == "get_reviews":
        for review in data.get("reviews", []):
            if review.get("id") and (review.get("text") or review.get("rating") is not None):
                units.add(("review", asin, str(review["id"])))
    elif tool == "get_product_details":
        for field, value in data.items():
            if _known(value) and value["value"] not in ("", [], {}):
                units.add(("field", asin, "detail", field))
    elif tool == "get_rating_info":
        for field in ("average_rating", "review_count"):
            if _known(data.get(field)):
                units.add(("field", asin, "rating", field))
    elif tool == "get_rank_info":
        for field in (
            "amazon_bsr",
            "case_bsr_order",
            "bsr_category",
            "simulated_case_rank",
            "simulated_purchase_count",
        ):
            v = data.get(field)
            if _known(v) or (type(v) in (int, float) and v != 0):
                units.add(("field", asin, "rank", field))
    elif tool == "get_price_info":
        # Initial shelf already contains the new-offer price; only genuinely new
        # used/list offers and returned history records count as query information.
        for field, value in (data.get("offers") or {}).items():
            if field != "new" and _known(value):
                units.add(("field", asin, "offers", field))
        for observation in data.get("observations", []):
            if (
                isinstance(observation, dict)
                and observation.get("at")
                and _known(observation.get("price"))
            ):
                units.add(("field", asin, "price_history", str(observation["at"])))
    # Sorting exposes no new market facts: the original shelf was already visible.
    return units


def compute_reward(trajectory, decision, target, config=RewardConfig()):
    if not isinstance(decision, dict) or decision.get("purchase") is not True:
        raise ValueError("valid completed Decision required")
    if not isinstance(target, dict) or not target.get("asin"):
        raise ValueError("target event required after Decision")
    seen = set()
    items = []
    cost = 0.0
    for step in trajectory:
        if step.get("tool") == "STOP":
            continue
        cost += float(step.get("cost", 0))
        units = information_units(step)
        new = units - seen
        if step.get("tool") == "get_reviews":
            new = set(sorted(new)[: config.max_reviews_per_page])
        seen.update(units)
        items.append(
            {
                "tool": step.get("tool"),
                "asin": step.get("asin"),
                "new_information_units": len(new),
                "cost": step.get("cost", 0),
            }
        )
    raw = sum(x["new_information_units"] for x in items)
    info = min(raw, config.max_information)
    match = int(decision.get("product_id") == target["asin"])
    quality = None
    rating = target.get("rating")
    probabilities = decision.get("rating_probabilities")
    if (
        config.scheme == "C"
        and match
        and type(rating) in (int, float)
        and rating in (1, 2, 3, 4, 5)
        and isinstance(probabilities, list)
        and len(probabilities) == 5
    ):
        import math

        if all(
            type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in probabilities
        ) and math.isclose(sum(probabilities), 1, abs_tol=1e-6):
            quality = 1 - abs(sum((i + 1) * p for i, p in enumerate(probabilities)) - rating) / 4
    info_term = config.alpha * info if config.scheme == "B" else 0.0
    quality_term = config.rating_weight * quality if quality is not None else 0.0
    return {
        "scheme": config.scheme,
        "decision_match": match,
        "information_units": info,
        "raw_information_units": raw,
        "query_cost": cost,
        "rating_quality": quality,
        "rating_quality_eligible": quality is not None,
        "information_term": info_term,
        "rating_quality_term": quality_term,
        "alpha": config.alpha,
        "beta": config.beta,
        "information_cap": config.max_information,
        "reward": match + info_term + quality_term - config.beta * cost,
        "queries": items,
        "config": asdict(config),
    }


def centered_advantages(rewards):
    """Return the mean baseline and centered advantages; reject constant batches."""
    import math, statistics

    values = [float(x) for x in rewards]
    if len(values) < 2 or not all(math.isfinite(x) for x in values):
        raise ValueError("at least two finite rewards required")
    if max(values) - min(values) < 1e-12:
        raise ValueError("constant rewards: no differentiated batch update")
    baseline = statistics.mean(values)
    return baseline, [x - baseline for x in values]
