"""Evidence-backed natural-language consumer profiles for neural Attention."""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

PROMPT_VERSION = "profile_v10_value_evidence"
PROFILE_SYSTEM = """You are a careful consumer-behavior researcher. Read ALL supplied pre-t0 review events in full, including text, rating, date/timestamp, product ID, and later updates. They are reviews and ratings, not independently verified purchases.

Your task is to write a natural-language profile of the REVIEWER'S recurring evaluation criteria and decision process, not a summary of the reviewed products. First reason across the entire history: compare reviews, identify repeated criteria, conflicting evidence, four-star or negative experiences, troubleshooting, revisions after new evidence, comparisons of alternatives, tradeoffs, and how the reviewer uses technical claims versus actual performance. Distinguish (a) patterns of evaluating or choosing that may transfer across categories from (b) domain-specific knowledge and tastes. A product feature or favorable experience is not automatically a stable preference. Do not carry audio-specific tastes into an unrelated product category.

Apply strict evidence calibration. Mentioning a brand or comparing brands does not establish brand loyalty; praising one plug-and-play installation does not establish a general preference for easy installation when the reviewer also accepts substantial modifications. A high rating does not erase criticism in the same review. If evidence is limited to one domain, say so. If price sensitivity, loyalty, novelty seeking, convenience preference, or another tendency is unsupported or mixed, put it in uncertainties instead of asserting it. Do not infer age, income, occupation, family, or other demographics.

The main output is narrative: write multiple substantial paragraphs, each with a distinct purpose. Explain supported recurring evaluation criteria and their evidence; explain practical decision, comparison, installation, and correction behavior where supported; describe domain-specific expertise separately; conclude with evidence limits and uncertainties. Mention concrete examples without turning the profile into a product-by-product recap. Length should reflect the information in the history, not an arbitrary word count.

Return one JSON object with exactly: user_id (string), narrative (string with paragraph breaks), preferences (array of supporting objects with aspect, description, evidence_review_ids), uncertainties (array of explicit strings). The number of preferences is evidence-driven, including zero if needed. Each preference must describe a reviewer tendency rather than a product attribute, and cite direct supplied review_id values. Check that the narrative and preferences do not contradict the negative or mixed reviews. Return valid JSON only."""
CHUNK_SYSTEM = """You are preparing evidence notes for a natural-language consumer profile. Read every supplied review event in this chunk, including full text, rating, timestamp, and product identifier. Record cross-review patterns and counterevidence in this chunk; distinguish consumer evaluation behavior from product attributes and domain-specific facts. Preserve positive, negative, mixed, and revised judgments. Do not infer demographics or verified purchases.

Return exactly one JSON object with chunk_narrative (substantive evidence-based prose), patterns (array of objects with observation, evidence_review_ids), counterevidence (array of objects with observation, evidence_review_ids), and uncertainties (array of strings). Do not limit the number of patterns. Cite only review_id values in this chunk. Return valid JSON only."""
MERGE_SYSTEM = """You are a consumer-behavior researcher. Merge all supplied chunk notes into a natural-language consumer profile. Account for every chunk, synthesize repeated patterns across the entire history, and preserve conflicts and uncertainty. Distinguish transferable evaluation and decision habits from domain-specific product experience. Do not turn a product's positive attribute into a general preference. Do not infer demographics or verified purchases.

Write a substantive multi-paragraph narrative as the primary semantic content, including recurring criteria, evaluation and decision methods, domain experience, and uncertainty when supported. Do not impose a fixed number of preferences or a fixed word count. Return exactly one JSON object with user_id (string), narrative (string), preferences (array of objects with aspect, description, evidence_review_ids), and uncertainties (array of strings). Preferences support the narrative. Cite only supplied review_id values. Return valid JSON only."""
REVIEW_SYSTEM = """You are an independent evidence auditor, not the author of the supplied draft. Re-read ALL supplied source reviews, ratings, timestamps and later updates. Produce an audit of the draft, not a replacement profile.

Check every claim against actual evidence. Internally audit every supplied review_id before writing; do not let long, vivid reviews eclipse shorter ones. The payload highlights source reviews not referenced in the draft; read them especially carefully and incorporate any distinct decision or evaluation pattern they support. Do not force irrelevant details into the narrative, but do not overlook a material theme merely because the draft missed it. A four-star review is not a low rating; criticism can coexist with a high rating. A single convenient installation does not establish a general preference for convenience when other reviews show willingness to make difficult modifications. Comparing or using brands is not automatically loyalty. Product audio qualities are domain-specific. Assess, without presupposing any conclusion, whether the source supports patterns in how the reviewer evaluates actual versus claimed performance, tests compatibility, handles installation and fault diagnosis, compares old and new products or alternative configurations, judges materials and cost relative to quality, and revises judgments. If reviews discuss material specifications, durability, price/value tradeoffs, or an alternative assembled at similar cost, do not omit that evidence merely because other reviews focus on audio performance. Keep unsupported claims in uncertainties, including claims about preferences outside the reviewed domain.

Return one JSON object with exactly: omissions (array of objects with observation and evidence_review_ids), overclaims (array of strings), corrections (array of strings), and uncertainties (array of strings). In omissions, note distinct evaluation or decision patterns missing from the draft, especially when unreferenced reviews support them. Do not write the final narrative. Every evidence_review_ids value must appear in the source reviews. Do not infer demographics or verified purchases. Return valid JSON only."""
FINALIZE_SYSTEM = """You are writing the final natural-language consumer profile after an independent evidence audit. Read the full draft, audit, and supplied reviews or chunk notes. Integrate the audit's omissions and corrections into a complete profile; do not output only the audit findings or a short addendum. The narrative is the primary semantic content. It should contain substantive paragraphs on recurring evaluation criteria and decision habits, practical testing/comparison/troubleshooting and revision, domain-specific knowledge, and uncertainty where supported by the source. Include material-quality and value tradeoffs if the evidence supports them. Distinguish an observed comparison of better materials at similar or lower cost from an unsupported broad claim of price sensitivity. Do not say there is no evidence of concern for value if any source review explicitly compares cost and quality. Separate domain-specific audio qualities from potentially transferable methods. Do not infer brand loyalty, general convenience preference, or price sensitivity from isolated evidence. Do not infer demographics or verified purchases. Avoid fixed numbers of preferences or words.

Return exactly one JSON object with user_id, narrative, preferences (array of objects with aspect, description, evidence_review_ids), and uncertainties (array of strings). Preserve only evidence IDs from the source. Return valid JSON only."""
CONSISTENCY_SYSTEM = """You are the final evidence and consistency editor for a natural-language consumer profile. Read the complete candidate profile and ALL supplied source reviews or chunk notes. Return a revised COMPLETE profile, preserving its substantive multi-paragraph coverage, not a short correction.

Check the narrative, preferences, and uncertainties against one another. Remove any assertion of a stable or quantified tendency when another field says evidence is insufficient. The payload separately highlights source reviews containing cost or material terms; these are still raw reviews, not predetermined conclusions. Read them closely. If a source review says the reviewer assembled a better-material alternative for about the same or less money, explain that concrete comparison in the narrative and cite its review_id; do not turn one comparison into global price sensitivity. Do not state there is no cost concern when such a comparison exists. Check that product audio qualities are identified as domain-specific, that installation convenience and brand loyalty are not overclaimed, and that a review update is described accurately. Preserve discussion of technical performance, compatibility, hands-on installation, fault correction, alternative comparisons, materials, and evidence limits only insofar as the source supports each. Do not infer demographics or verified purchases.

Return exactly one JSON object with user_id, narrative (several substantive paragraphs if the history warrants them), preferences (array of objects with aspect, description, evidence_review_ids), and uncertainties (array of strings). Cite only supplied review_id values. Return valid JSON only."""


def history_for_user(case_store, user_id):
    cutoff = case_store.t0.timestamp() * 1000
    return [
        dict(r)
        for r in case_store.users.get(user_id, [])
        if r.get("user_id") == user_id
        and isinstance(r.get("timestamp_ms"), (int, float))
        and r["timestamp_ms"] < cutoff
    ]


def validate_profile(profile, history):
    if not history:
        raise ValueError("profile requires pre-t0 history")
    if not isinstance(profile, dict) or set(profile) != {
        "user_id",
        "narrative",
        "preferences",
        "uncertainties",
    }:
        raise ValueError("invalid profile keys")
    if (
        profile["user_id"] != history[0].get("user_id")
        or not isinstance(profile["narrative"], str)
        or not profile["narrative"].strip()
        or not isinstance(profile["preferences"], list)
        or not isinstance(profile["uncertainties"], list)
        or not all(isinstance(x, str) for x in profile["uncertainties"])
    ):
        raise ValueError("invalid profile fields")
    ids = {r.get("review_id") for r in history}
    for p in profile["preferences"]:
        if (
            not isinstance(p, dict)
            or set(p) != {"aspect", "description", "evidence_review_ids"}
            or not isinstance(p["aspect"], str)
            or not isinstance(p["description"], str)
            or not isinstance(p["evidence_review_ids"], list)
            or not all(isinstance(rid, str) for rid in p["evidence_review_ids"])
            or not set(p["evidence_review_ids"]) <= ids
        ):
            raise ValueError("invalid preference evidence")
    return profile


class ProfileExtractor:
    def __init__(self, client, cache_dir, model_id, chunk_size=20, repair_invalid=False):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.client = client
        self.cache = Path(cache_dir)
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.repair_invalid = repair_invalid
        self.cache.mkdir(parents=True, exist_ok=True)

    def extract(self, user_id, history):
        if not history:
            raise ValueError("profile requires pre-t0 history")
        digest = hashlib.sha256(
            json.dumps(history, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        path = (
            self.cache
            / f"{user_id}_{digest}_{hashlib.sha256((self.model_id + PROMPT_VERSION + str(self.chunk_size)).encode()).hexdigest()}.json"
        )
        if path.exists():
            profile = validate_profile(json.loads(path.read_text(encoding="utf-8")), history)
            if hasattr(self.client, "record_cache_hit"):
                self.client.record_cache_hit("Profile Extraction", self.model_id)
            return profile
        notes = []
        if len(history) <= self.chunk_size:
            profile, _ = self.client.complete_json(
                "profile",
                PROFILE_SYSTEM,
                {"user_id": user_id, "reviews": history, "prompt_version": PROMPT_VERSION},
                agent_id=user_id,
                max_tokens=3500,
                stage="Profile Extraction",
            )
        else:
            for offset in range(0, len(history), self.chunk_size):
                reviews = history[offset : offset + self.chunk_size]
                note, _ = self.client.complete_json(
                    "profile",
                    CHUNK_SYSTEM,
                    {
                        "user_id": user_id,
                        "chunk_index": offset // self.chunk_size,
                        "reviews": reviews,
                        "prompt_version": PROMPT_VERSION,
                    },
                    agent_id=user_id,
                    max_tokens=2200,
                    stage="Profile Extraction",
                )
                notes.append(note)
            profile, _ = self.client.complete_json(
                "profile",
                MERGE_SYSTEM,
                {
                    "user_id": user_id,
                    "notes": notes,
                    "all_review_ids": [r["review_id"] for r in history],
                    "prompt_version": PROMPT_VERSION,
                },
                agent_id=user_id,
                max_tokens=3500,
                stage="Profile Merge",
            )
        draft_narrative = profile["narrative"]
        draft_text = json.dumps(profile, ensure_ascii=False)
        unreferenced_reviews = [r for r in history if r["review_id"] not in draft_text]
        audit, _ = self.client.complete_json(
            "profile",
            REVIEW_SYSTEM,
            {
                "user_id": user_id,
                "draft": profile,
                "reviews": history if len(history) <= self.chunk_size else None,
                "chunk_notes": notes if len(history) > self.chunk_size else None,
                "unreferenced_reviews": unreferenced_reviews,
                "prompt_version": PROMPT_VERSION,
            },
            agent_id=user_id,
            max_tokens=2500,
            stage="Profile Review",
        )
        profile, _ = self.client.complete_json(
            "profile",
            FINALIZE_SYSTEM,
            {
                "user_id": user_id,
                "draft": profile,
                "audit": audit,
                "reviews": history if len(history) <= self.chunk_size else None,
                "chunk_notes": notes if len(history) > self.chunk_size else None,
                "prompt_version": PROMPT_VERSION,
            },
            agent_id=user_id,
            max_tokens=4200,
            stage="Profile Finalization",
        )
        cost_material_evidence = [
            r
            for r in history
            if re.search(
                r"\b(cost|price|paid|payed|less|material|value|worth)\b",
                str(r.get("text") or ""),
                re.I,
            )
        ]
        profile, _ = self.client.complete_json(
            "profile",
            CONSISTENCY_SYSTEM,
            {
                "user_id": user_id,
                "candidate": profile,
                "reviews": history if len(history) <= self.chunk_size else None,
                "chunk_notes": notes if len(history) > self.chunk_size else None,
                "cost_material_evidence": cost_material_evidence,
                "prompt_version": PROMPT_VERSION,
            },
            agent_id=user_id,
            max_tokens=4200,
            stage="Profile Consistency",
        )

        def validate_final(candidate):
            validate_profile(candidate, history)
            if len(history) >= 5 and (
                len(candidate["narrative"].split("\n\n")) < 2
                or len(candidate["narrative"]) < len(draft_narrative) * 0.6
            ):
                raise ValueError("narrative too brief for the supplied history")

        try:
            validate_final(profile)
        except ValueError as exc:
            if not self.repair_invalid:
                raise
            repair_system = "Return the complete corrected profile JSON. Preserve the substantive multi-paragraph narrative and preferences; repair invalid schema or evidence IDs using only supplied source reviews. Omit unsupported preferences. Do not invent evidence."
            profile, _ = self.client.complete_json(
                "profile",
                repair_system,
                {
                    "user_id": user_id,
                    "candidate": profile,
                    "draft": draft_narrative,
                    "reviews": history,
                    "validation_error": str(exc),
                },
                agent_id=user_id,
                max_tokens=4200,
                stage="Profile Repair",
            )
            try:
                validate_final(profile)
            except ValueError as second:
                exact_system = "Repair the supplied consumer profile to exactly four JSON keys: user_id, narrative, preferences, uncertainties. Each preference must have exactly aspect, description, evidence_review_ids. Preserve substantive narrative. Cite only actual supplied review_id values and omit unsupported claims. Do not add keys."
                profile, _ = self.client.complete_json(
                    "profile",
                    exact_system,
                    {
                        "user_id": user_id,
                        "candidate": profile,
                        "reviews": history,
                        "validation_error": str(second),
                    },
                    agent_id=user_id,
                    max_tokens=6500,
                    stage="Profile Repair",
                )
                validate_final(profile)
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        return profile


def profile_input(profile, history, embeddings):
    ratings = [float(r["rating"]) for r in history if isinstance(r.get("rating"), (float, int))]
    timestamps = [r["timestamp_ms"] for r in history]
    texts = [str(r.get("text") or "") for r in history]
    features = [
        min(math.log1p(len(history)) / 8, 1),
        min(math.log1p(len({r.get("asin") for r in history})) / 8, 1),
        sum(ratings) / len(ratings) / 5 if ratings else 0,
        float(bool(ratings)),
        min(np.mean([len(t) for t in texts]) / 2000, 1) if texts else 0,
        float(bool(texts)),
        min((max(timestamps) - min(timestamps)) / 1000 / 86400 / 3650, 1) if timestamps else 0,
        float(bool(timestamps)),
    ]
    return np.concatenate(
        [
            embeddings.encode(profile["narrative"]),
            embeddings.mean(texts),
            np.asarray(features, dtype="float32"),
        ]
    ).astype("float32")
