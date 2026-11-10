"""Explicit review-event-conditioned evaluation; not population purchase estimates."""

import math
from collections import Counter
from ..protocol import validate_review_event

NAMES = (
    "JSD_full",
    "TVD",
    "selection_rate_error",
    "NDCG@5",
    "MAE",
    "W1",
    "sentiment_consistency",
    "product_factual_consistency",
    "content_alignment",
)


def ndcg5_ties(truth_counts, predicted_counts, products):
    order = sorted(products, key=lambda p: -predicted_counts.get(p, 0))
    dcg = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and predicted_counts.get(order[j], 0) == predicted_counts.get(
            order[i], 0
        ):
            j += 1
        # Average full tie group's discounts, including zero beyond cutoff.
        discount = sum(1 / math.log2(k + 2) if k < 5 else 0 for k in range(i, j)) / (j - i)
        dcg += sum(truth_counts.get(p, 0) for p in order[i:j]) * discount
        i = j
    ideal = sorted((truth_counts.get(p, 0) for p in products), reverse=True)
    denominator = sum(v / math.log2(i + 2) for i, v in enumerate(ideal[:5]))
    return dcg / denominator if denominator else None


def evaluate_review_events(
    result, truth, products, *, judge=None, t0_facts=None, judge_max=0, unit="user"
):
    if unit not in ("user", "event"):
        raise ValueError("unit must be user or event")
    products = tuple(products)
    if not products or len(set(products)) != len(products):
        raise ValueError("products must be unique and nonempty")
    rows = [x for d in result["days"] for x in d["sessions"]]
    if len({r["session_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sessions")
    missing_simulation_events = set(truth) - {r["session_id"] for r in rows}
    if unit == "user" and len({r["user_id"] for r in rows}) != len(rows):
        raise ValueError("select one predetermined event per user for user-level metrics")
    pairs = []
    errors = []
    for row in rows:
        target = truth.get(row["session_id"])
        if (
            not target
            or target.get("user_id") != row["user_id"]
            or target.get("timestamp_ms") != row["scheduled_at_ms"]
            or target.get("asin") not in products
        ):
            errors.append(
                {"session_id": row["session_id"], "reason": "missing_or_mismatched_truth"}
            )
            continue
        d = row.get("decision") or {}
        try:
            if row.get("status") != "ok":
                raise ValueError("failed_session")
            clean, _ = validate_review_event(
                {
                    "selected_asin": d["product_id"],
                    "rating_probabilities": d["rating_probabilities"],
                    "rating": d["rating"],
                    "review_text": d["review_text"],
                },
                set(products),
            )
        except (KeyError, ValueError, TypeError):
            errors.append({"session_id": row["session_id"], "reason": "invalid_decision"})
            continue
        pairs.append((row, target, clean))
    n = len(pairs)
    truth_counts = Counter(t["asin"] for r, t, d in pairs)
    sim_counts = Counter(d["product_id"] for r, t, d in pairs)

    def metric(value, status="ok", sample=None, note=None):
        return {
            "value": value,
            "status": status,
            "sample_size": n if sample is None else sample,
            "note": note,
        }

    metrics = {}
    note = (
        "Review-event-conditioned matched users"
        if unit == "user"
        else "Review-event-conditioned scheduled events; repeated consumers counted per event"
    ) + "; NONE is zero by protocol; not population purchase selection."
    if n and n == len(rows) and not missing_simulation_events:
        p = [truth_counts[x] / n for x in products] + [0.0]
        q = [sim_counts[x] / n for x in products] + [0.0]
        mid = [(a + b) / 2 for a, b in zip(p, q)]
        js = 0.5 * sum(a * math.log2(a / m) for a, m in zip(p, mid) if a) + 0.5 * sum(
            b * math.log2(b / m) for b, m in zip(q, mid) if b
        )
        metrics["JSD_full"] = metric(js, note=note)
        metrics["TVD"] = metric(0.5 * sum(abs(a - b) for a, b in zip(p, q)), note=note)
        metrics["selection_rate_error"] = metric(abs(p[-1] - q[-1]), note=note)
        metrics["NDCG@5"] = metric(ndcg5_ties(truth_counts, sim_counts, products), note=note)
    else:
        for name in NAMES[:4]:
            metrics[name] = metric(
                None,
                "incomplete_cohort" if n else "insufficient_data",
                n,
                "Fixed-user distribution unavailable: one or more scheduled events, decisions or targets are missing; failed sessions are not NONE.",
            )
    errors_rating = []
    w = []
    for row, t, d in pairs:
        actual = t.get("rating")
        prob = d["rating_probabilities"]
        if type(actual) not in (int, float) or actual not in (1, 2, 3, 4, 5):
            continue
        mean = sum((i + 1) * v for i, v in enumerate(prob))
        errors_rating.append(abs(mean - actual))
        w.append(sum(abs(sum(prob[:s]) - float(actual <= s)) for s in range(1, 5)))
    for name, values in [("MAE", errors_rating), ("W1", w)]:
        metrics[name] = metric(
            sum(values) / len(values) if values else None,
            "ok" if values else "insufficient_data",
            len(values),
            "Event-paired regardless of predicted product; uses full predicted rating distribution.",
        )
    judge_scores = []
    eligible = []
    for row, t, d in pairs:
        if d["product_id"] == t["asin"] and isinstance(t.get("text"), str) and t["text"].strip():
            eligible.append((row, t, d))
    if judge is not None and judge_max > 0:
        for row, t, d in eligible[:judge_max]:
            payload = {
                "generated_review": d["review_text"],
                "real_review": t["text"],
                "real_rating": t.get("rating"),
                "t0_product_information": (t0_facts or {}).get(t["asin"]),
            }
            try:
                judge_scores.append(judge.score(payload))
            except Exception:
                continue
    for name in NAMES[6:]:
        values = [x[name] / 4 for x in judge_scores if x.get(name) is not None]
        status = (
            "ok"
            if values
            else ("not_evaluated" if judge is None or judge_max <= 0 else "insufficient_data")
        )
        metrics[name] = metric(
            sum(values) / len(values) if values else None,
            status,
            len(values),
            "Exact same event and product, Judge 0–4 normalized by 4; eligible pairs="
            + str(len(eligible)),
        )
    return {
        "schema": "review_event_evaluation_v1",
        "case_id": result.get("case_id"),
        "seed": result.get("seed"),
        "unit": unit,
        "metrics": metrics,
        "coverage": {
            "scheduled_users": len(rows),
            "valid_users": n,
            "failed_or_unmatched": len(errors),
            "event_level_rows": len(rows),
            "rating_pairs": len(errors_rating),
            "product_exact_matches": sum(d["product_id"] == t["asin"] for r, t, d in pairs),
            "truth_events": len(truth),
            "missing_simulation_events": len(missing_simulation_events),
        },
        "counts": {
            "truth": dict(truth_counts),
            "predicted": dict(sim_counts),
            "none_truth": 0,
            "none_predicted": 0,
        },
        "diagnostics": errors,
        "comparability": "conditional review-event proxy; complete paired cases only; not full population, no Judge",
    }
