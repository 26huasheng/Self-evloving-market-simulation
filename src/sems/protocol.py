"""Decision contracts shared by inference and offline evaluation (standard library only)."""

import math


def validate_decision(value, shelf):
    if not isinstance(value, dict):
        raise ValueError("decision must be a JSON object")
    required = {
        "purchase",
        "product_id",
        "submit_rating",
        "rating",
        "publish_review",
        "review_text",
        "rationale",
    }
    if set(value) != required:
        raise ValueError("decision fields do not match protocol")
    for name in ("purchase", "submit_rating", "publish_review"):
        if type(value[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if not isinstance(value["review_text"], str) or not isinstance(value["rationale"], str):
        raise ValueError("review_text and rationale must be strings")
    if not value["purchase"]:
        if (
            value["product_id"] is not None
            or value["submit_rating"]
            or value["publish_review"]
            or value["rating"] is not None
            or value["review_text"]
        ):
            raise ValueError("non-purchase cannot include product, rating or review")
        return []
    asin = value["product_id"]
    if asin not in shelf:
        raise ValueError("product_id outside visible shelf")
    if value["submit_rating"]:
        if type(value["rating"]) is not int or value["rating"] not in range(1, 6):
            raise ValueError("rating must be an integer from 1 to 5")
    elif value["rating"] is not None:
        raise ValueError("rating supplied without submit_rating")
    if value["publish_review"]:
        if not value["review_text"].strip():
            raise ValueError("review text is empty")
    elif value["review_text"]:
        raise ValueError("review text supplied without publish_review")
    actions = [{"tool": "purchase_product", "args": {"asin": asin}}]
    if value["submit_rating"]:
        actions.append({"tool": "rate_product", "args": {"asin": asin, "rating": value["rating"]}})
    if value["publish_review"]:
        actions.append(
            {"tool": "publish_review", "args": {"asin": asin, "text": value["review_text"]}}
        )
    return actions


def validate_review_event(value, shelf):
    if not isinstance(value, dict) or set(value) != {
        "selected_asin",
        "rating_probabilities",
        "rating",
        "review_text",
    }:
        raise ValueError("expected selected_asin, rating_probabilities, rating, review_text")
    if value["selected_asin"] not in shelf:
        raise ValueError("selected_asin outside visible shelf")
    probabilities = value["rating_probabilities"]
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != 5
        or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in probabilities)
        or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
    ):
        raise ValueError(
            "rating_probabilities must contain five finite nonnegative values summing to one"
        )
    if type(value["rating"]) is not int or value["rating"] not in range(1, 6):
        raise ValueError("rating must be an integer from 1 to 5")
    if not isinstance(value["review_text"], str) or not value["review_text"].strip():
        raise ValueError("review_text must be nonempty")
    decision = {
        "purchase": True,
        "product_id": value["selected_asin"],
        "submit_rating": True,
        "rating": value["rating"],
        "rating_probabilities": probabilities,
        "publish_review": True,
        "review_text": value["review_text"],
        "rationale": "",
    }
    actions = validate_decision(
        {k: v for k, v in decision.items() if k != "rating_probabilities"}, shelf
    )
    return decision, actions
