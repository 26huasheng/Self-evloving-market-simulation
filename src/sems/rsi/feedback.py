"""Post-decision outcome reward; no target reaches the policy or Decision."""

import math
import numpy as np


def outcome_reward(
    decision,
    target,
    product_titles,
    embeddings,
    query_cost,
    *,
    similarity_weight=0.2,
    rating_weight=0.1,
    cost_weight=0.001,
):
    if not isinstance(decision, dict) or decision.get("purchase") is not True:
        raise ValueError("completed decision required")
    selected = decision["product_id"]
    actual = target["asin"]
    if selected not in product_titles or actual not in product_titles:
        raise ValueError("unknown product")
    a = np.asarray(embeddings.encode(product_titles[selected]), dtype="float64")
    b = np.asarray(embeddings.encode(product_titles[actual]), dtype="float64")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    similarity = max(0.0, min(1.0, float(np.dot(a, b) / denom))) if denom else 0.0
    match = int(selected == actual)
    rating = target.get("rating")
    probs = decision.get("rating_probabilities")
    quality = None
    if (
        match
        and type(rating) in (int, float)
        and rating in (1, 2, 3, 4, 5)
        and isinstance(probs, list)
        and len(probs) == 5
        and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in probs)
        and math.isclose(sum(probs), 1, abs_tol=1e-6)
    ):
        quality = 1 - abs(sum((i + 1) * x for i, x in enumerate(probs)) - rating) / 4
    cost = float(query_cost)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("invalid cost")
    reward = (
        match
        + similarity_weight * similarity
        + (rating_weight * quality if quality is not None else 0)
        - cost_weight * cost
    )
    return {
        "reward": reward,
        "exact_match": match,
        "title_similarity": similarity,
        "similarity_weight": similarity_weight,
        "matched_rating_quality": quality,
        "rating_weight": rating_weight,
        "paid_query_cost": cost,
        "cost_weight": cost_weight,
        "note": "Title similarity is a weak post-decision proxy, not a true market outcome or consumer preference.",
    }
