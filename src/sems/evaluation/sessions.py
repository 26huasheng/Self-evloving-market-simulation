"""Session-level evaluation; future truth is consumed only in this module's caller."""

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

JUDGE_VERSION = "review_judge_v1"
JUDGE_SYSTEM = """You are an impartial review evaluator. Compare only the supplied matched reviews for the same consumer event and product. Score sentiment_consistency and content_alignment on integer 1-5: 1 means strong conflict, 3 partial or unclear alignment, 5 close alignment. Score product_factual_consistency on the same scale only when verifiable facts are supplied. Distinguish unsupported statements from contradicted statements: lack of evidence is not an error; if essential claims cannot be verified, return null for factual score. Return exactly JSON with sentiment_consistency, product_factual_consistency, content_alignment (each integer 1-5 or null), and reasons (object with a short string for each metric). Do not infer product facts from the real review. No markdown."""
JUDGE_NAMES = ("sentiment_consistency", "product_factual_consistency", "content_alignment")


def metric(
    name,
    value,
    status,
    definition,
    sample_size,
    comparability,
    coverage=None,
    direction="down",
    note=None,
):
    return {
        "name": name,
        "value": value,
        "status": status,
        "definition": definition,
        "sample_size": sample_size,
        "comparability": comparability,
        "coverage": coverage,
        "direction": direction,
        "note": note,
    }


def distribution(counts, products):
    n = sum(counts.values())
    return {p: counts.get(p, 0) / n for p in products} if n else None


def jsd(p, q):
    def kl(a, b):
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)

    mid = [(x + y) / 2 for x, y in zip(p, q)]
    return (kl(p, mid) + kl(q, mid)) / 2


def tvd(p, q):
    return sum(abs(x - y) for x, y in zip(p, q)) / 2


def ndcg5(true_counts, sim_counts, products):
    order = sorted(products, key=lambda p: (-sim_counts.get(p, 0), p))[:5]
    ideal = sorted(products, key=lambda p: (-true_counts.get(p, 0), p))[:5]

    def dcg(items):
        return sum((2 ** true_counts.get(p, 0) - 1) / math.log2(i + 2) for i, p in enumerate(items))

    denominator = dcg(ideal)
    return dcg(order) / denominator if denominator else None


def w1(real, sim):
    return sum(
        abs(sum(x <= k for x in real) / len(real) - sum(x <= k for x in sim) / len(sim))
        for k in range(1, 5)
    )


def valid_rating(value):
    return type(value) in (int, float) and value in (1, 2, 3, 4, 5)


class Judge:
    """Explicitly configured client; cache key includes model, protocol and payload."""

    def __init__(self, client, model, cache_dir):
        self.client = client
        self.model = model
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.hits = 0

    def score(self, payload):
        key = hashlib.sha256(
            json.dumps(
                {
                    "version": JUDGE_VERSION,
                    "system": JUDGE_SYSTEM,
                    "model": self.model,
                    "payload": payload,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        path = self.cache / (key + ".json")
        if path.exists():
            self.hits += 1
            if hasattr(self.client, "record_cache_hit"):
                self.client.record_cache_hit("Judge", self.model)
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            result, _ = self.client.complete_json(
                "judge", JUDGE_SYSTEM, payload, stage="Judge", max_tokens=700
            )
            self._validate(result)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self._validate(result)
        return result

    @staticmethod
    def _validate(result):
        if not isinstance(result, dict) or set(result) != set(JUDGE_NAMES) | {"reasons"}:
            raise ValueError("invalid judge fields")
        if (
            not isinstance(result["reasons"], dict)
            or set(result["reasons"]) != set(JUDGE_NAMES)
            or not all(isinstance(x, str) for x in result["reasons"].values())
        ):
            raise ValueError("invalid judge reasons")
        for name in JUDGE_NAMES:
            value = result[name]
            if value is not None and (type(value) is not int or not 1 <= value <= 5):
                raise ValueError("judge scores must be integers 1-5 or null")


def evaluate(
    result, truth, products, *, judge=None, judge_max=0, seed=0, product_facts=None, api_usage=None
):
    if type(judge_max) is not int or judge_max < 0:
        raise ValueError("judge_max must be nonnegative")
    products = tuple(products)
    if not products or len(set(products)) != len(products):
        raise ValueError("product universe must be unique and nonempty")
    rows = [s for day in result["days"] for s in day["sessions"]]
    if len({s["session_id"] for s in rows}) != len(rows):
        raise ValueError("duplicate session IDs")
    matched = []
    diagnostics = []
    for s in rows:
        t = truth.get(s["session_id"])
        reason = None
        if t is None:
            reason = "missing_truth"
        elif t.get("user_id") != s.get("user_id") or t.get("timestamp_ms") != s.get(
            "scheduled_at_ms"
        ):
            reason = "event_mismatch"
        elif t.get("asin") not in products:
            reason = "truth_outside_shelf"
        elif s.get("status") != "ok":
            reason = "simulation_failed"
        elif not isinstance(s.get("decision"), dict):
            reason = "missing_decision"
        elif s["decision"].get("purchase") and s["decision"].get("product_id") not in products:
            reason = "simulation_outside_shelf"
        if reason:
            diagnostics.append(
                {"session_id": s["session_id"], "user_id": s.get("user_id"), "status": reason}
            )
            continue
        matched.append((s, t))
    true_counts = Counter(t["asin"] for s, t in matched)
    all_true_counts = Counter(t["asin"] for t in truth.values() if t.get("asin") in products)
    purchases = Counter(
        s["decision"]["product_id"] for s, t in matched if s["decision"].get("purchase") is True
    )
    comments = Counter(
        s["decision"]["product_id"]
        for s, t in matched
        if s["decision"].get("purchase") is True
        and s["decision"].get("publish_review") is True
        and s["decision"].get("review_text", "").strip()
    )
    true_dist = distribution(true_counts, products)
    sim_dist = distribution(comments, products)
    cohort = len(matched)
    total = len(truth)
    coverage = cohort / total if total else None
    comp = "proxy: completed scheduled review events; real review targets vs simulated published-review targets; neither is a complete purchase distribution"
    available = bool(true_dist and sim_dist)
    p = [true_dist[x] for x in products] if true_dist else []
    q = [sim_dist[x] for x in products] if sim_dist else []
    metrics = {}
    metrics["JSD_full"] = metric(
        "JSD_full",
        None,
        "not_identifiable",
        "Jensen-Shannon divergence over products plus no-selection",
        {"completed_events": cohort},
        "baseline definition and real no-selection fraction unavailable",
        coverage,
        note="A review-conditioned cohort contains no observed non-selection denominator.",
    )
    metrics["TVD"] = metric(
        "TVD",
        tvd(p, q) if available else None,
        "ok" if available else "insufficient_data",
        "0.5 × sum over shelf products of |P(real review target)-P(simulated review target)|, each conditioned on a review",
        {"real_reviews": sum(true_counts.values()), "sim_reviews": sum(comments.values())},
        comp,
        coverage,
    )
    n = ndcg5(true_counts, comments, products) if available else None
    metrics["NDCG@5"] = metric(
        "NDCG@5",
        n,
        "ok" if n is not None else "insufficient_data",
        "DCG@5 = sum((2^real review count - 1)/log2(rank+1)); rank by simulated review count descending, ties by ASIN; ideal rank by real count",
        {"real_reviews": sum(true_counts.values()), "sim_reviews": sum(comments.values())},
        comp,
        coverage,
        direction="up",
    )
    metrics["selection_rate_error"] = metric(
        "selection_rate_error",
        None,
        "not_identifiable",
        "absolute difference between real and simulated selection rates",
        {"completed_events": cohort},
        "real eligible population and no-selection outcomes unavailable",
        coverage,
    )
    rating_pairs = []
    rating_diagnostics = []
    review_candidates = []
    true_ratings = []
    sim_ratings = []
    for s, t in matched:
        d = s["decision"]
        tr = t.get("rating")
        sr = d.get("rating")
        same = d.get("purchase") is True and d.get("product_id") == t["asin"]
        generated = (
            d.get("purchase") is True
            and d.get("publish_review") is True
            and bool(d.get("review_text", "").strip())
        )
        if valid_rating(tr):
            true_ratings.append(float(tr))
        if generated and d.get("submit_rating") is True and valid_rating(sr):
            sim_ratings.append(float(sr))
        status = (
            "matched"
            if same and d.get("submit_rating") is True and valid_rating(tr) and valid_rating(sr)
            else ("product_mismatch" if d.get("purchase") and not same else "unpaired_or_unrated")
        )
        row = {
            "session_id": s["session_id"],
            "user_id": s["user_id"],
            "true_product": t["asin"],
            "sim_product": d.get("product_id"),
            "true_rating": tr,
            "sim_rating": sr if d.get("submit_rating") else None,
            "status": status,
            "absolute_error": abs(float(tr) - float(sr)) if status == "matched" else None,
        }
        rating_diagnostics.append(row)
        if status == "matched":
            rating_pairs.append(row["absolute_error"])
        if not generated:
            jstatus = "no_simulated_review"
        elif not same:
            jstatus = "product_mismatch"
        elif not isinstance(t.get("text"), str) or not t["text"].strip():
            jstatus = "missing_real_review"
        else:
            jstatus = "eligible"
            review_candidates.append((s, t))
        row["review_match_status"] = jstatus
    pair_rate = len(rating_pairs) / cohort if cohort else None
    metrics["MAE"] = metric(
        "MAE",
        sum(rating_pairs) / len(rating_pairs) if rating_pairs else None,
        "ok" if rating_pairs else "insufficient_data",
        "mean |true rating - simulated rating| on same session, consumer and product with both ratings",
        {"paired": len(rating_pairs), "completed_events": cohort},
        "matched product events only; selection bias",
        pair_rate,
    )
    metrics["W1"] = metric(
        "W1",
        w1(true_ratings, sim_ratings) if true_ratings and sim_ratings else None,
        "ok" if true_ratings and sim_ratings else "insufficient_data",
        "sum k=1..4 |F_real(k)-F_sim(k)| on 1–5 star distributions; simulated ratings require a published review",
        {"real_ratings": len(true_ratings), "sim_review_ratings": len(sim_ratings)},
        "unpaired completed-event distributions; product and review coverage may differ",
        len(sim_ratings) / cohort if cohort else None,
    )
    details = []
    candidates = sorted(review_candidates, key=lambda x: x[0]["session_id"])
    chosen = (
        random.Random(seed).sample(candidates, min(judge_max, len(candidates)))
        if judge is not None
        else []
    )
    for s, t in chosen:
        facts = (product_facts or {}).get(t["asin"])
        payload = {
            "session_id": s["session_id"],
            "product_id": t["asin"],
            "real_review": {"text": t["text"], "rating": t.get("rating")},
            "simulated_review": {
                "text": s["decision"]["review_text"],
                "rating": s["decision"].get("rating")
                if s["decision"].get("submit_rating")
                else None,
            },
            "verified_product_facts": facts or None,
        }
        try:
            scores = judge.score(payload)
            if not facts:
                scores = dict(scores, product_factual_consistency=None)
            details.append(
                {
                    "session_id": s["session_id"],
                    "user_id": s["user_id"],
                    "product_id": t["asin"],
                    "status": "scored",
                    "facts_available": bool(facts),
                    "scores": scores,
                }
            )
        except Exception as exc:
            details.append(
                {
                    "session_id": s["session_id"],
                    "user_id": s["user_id"],
                    "product_id": t["asin"],
                    "status": "judge_failed",
                    "error_type": type(exc).__name__,
                }
            )
    for name in JUDGE_NAMES:
        values = [
            x["scores"][name]
            for x in details
            if x["status"] == "scored" and x["scores"][name] is not None
        ]
        status = (
            "disabled"
            if judge is None or judge_max == 0
            else ("ok" if values else "insufficient_data")
        )
        metrics[name] = metric(
            name,
            sum(values) / len(values) if values else None,
            status,
            "custom Judge mean of valid scores on a 1–5 scale; same session/user/product; factual score requires independently verified product facts",
            {
                "eligible_pairs": len(candidates),
                "sampled": len(chosen),
                "valid": len(values),
                "missing": total - len(values),
            },
            "custom protocol; no established Baseline prompt or scale",
            len(values) / total if total else None,
            direction="up",
            note="Unknown facts are not evidence of an error.",
        )
    conditional = metric(
        "JSD_conditional_review_targets",
        jsd(p, q) if available else None,
        "ok" if available else "insufficient_data",
        "Jensen-Shannon divergence in bits over shelf products, conditioned on review",
        {"real_reviews": sum(true_counts.values()), "sim_reviews": sum(comments.values())},
        comp,
        coverage,
    )
    product_diagnostics = [
        {
            "product_id": a,
            "all_real_review_count": all_true_counts[a],
            "real_review_count": true_counts[a],
            "sim_purchase_count": purchases[a],
            "sim_review_count": comments[a],
            "real_review_share": true_dist[a] if true_dist else None,
            "sim_review_share": sim_dist[a] if sim_dist else None,
            "review_share_error": sim_dist[a] - true_dist[a] if available else None,
        }
        for a in products
    ]
    return {
        "schema": "session_evaluation_v1",
        "case_id": result.get("case_id"),
        "configuration": {
            "simulation_seed": result.get("seed"),
            "attention_model": result.get("attention_model"),
            "profile_version": result.get("profile_version"),
            "judge_enabled": judge is not None and judge_max > 0,
            "judge_model": judge.model if judge else None,
            "judge_version": JUDGE_VERSION,
            "judge_max": judge_max,
            "judge_seed": seed,
        },
        "metrics": metrics,
        "auxiliary_metrics": {conditional["name"]: conditional},
        "coverage": {
            "scheduled_sessions": total,
            "sessions_in_result": len(rows),
            "completed_matched_sessions": cohort,
            "decision_coverage": coverage,
            "paired_ratings": len(rating_pairs),
            "paired_rating_rate": pair_rate,
            "real_review_ratings": len(true_ratings),
            "simulated_review_ratings": len(sim_ratings),
            "same_product_review_pairs": len(candidates),
            "simulated_purchases": sum(purchases.values()),
            "simulated_reviews": sum(comments.values()),
        },
        "distributions": {
            "real_purchase": "unavailable",
            "all_real_review_targets": dict(all_true_counts),
            "real_review_targets_completed_cohort": dict(true_counts),
            "simulated_purchases": dict(purchases),
            "simulated_review_targets": dict(comments),
        },
        "diagnostics": {
            "products": product_diagnostics,
            "events": rating_diagnostics + diagnostics,
            "judge": details,
        },
        "api_usage": api_usage
        or {
            "actual_requests": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "cache_hits": 0,
            "input_tokens": None,
            "output_tokens": None,
        },
    }
