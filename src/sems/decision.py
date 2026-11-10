"""Decision-model boundary: configurable OpenAI-compatible API plus local dev model."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .api_client import SEMSAPIClient


class ChatClient(Protocol):
    mode: str

    def complete_json(self, system: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...


class OpenAICompatibleClient:
    """DecisionModel adapter over the single shared OpenAI SDK client."""

    mode = "api"

    def __init__(self, client: SEMSAPIClient | None = None):
        self.client = client or SEMSAPIClient()

    @classmethod
    def from_environment(cls):
        return cls(SEMSAPIClient())

    def complete_json(self, system: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        parsed, _ = self.client.complete_json(
            "decision",
            system,
            payload,
            agent_id=payload.get("agent_id"),
            evolution_round=int(payload.get("round", 0)),
            max_tokens=700,
            stage="Decision",
        )
        return parsed


PURCHASE_DECISION_SYSTEM = """You are a consumer deciding whether to purchase. Use only the supplied profile, visible shelf and obtained observations. Return a JSON object with purchase (boolean), product_id (visible shelf product ID if purchasing, otherwise null), submit_rating (boolean), rating (integer 1 to 5 if submitting a rating, otherwise null), publish_review (boolean), review_text (nonempty string if publishing a review, otherwise empty), and rationale (string). Ratings and reviews require a purchase. Do not request tools or invent observations."""


REVIEW_EVENT_SYSTEM = """You are a consumer in a review-event-conditioned market simulation. A purchase, star rating and review must occur. Use the complete supplied profile, initial and current visible shelf, and all supplied tool_results and read_reviews. Match each result to the ASIN in its action; a result with no product applies to the shelf. Treat unqueried, unknown, unavailable and not-applicable fields as unknown, not as negative evidence. Compare the visible alternatives using the profile and acquired evidence; shelf position alone is not evidence of preference. Never invent hidden product facts or request tools. Select exactly one product from the current visible shelf. Return ONLY JSON with exactly: selected_asin (visible product ID), rating_probabilities (five finite nonnegative numbers summing to 1, for stars 1 through 5), rating (integer 1 to 5), review_text (nonempty string)."""


class DecisionModel:
    INPUT_PROTOCOL_VERSION = "review_event_payload_dedup_v2"

    def __init__(self, client: ChatClient):
        self.client = client

    @property
    def mode(self):
        return self.client.mode

    def decide_review_event(self, agent_id, profile, observation, correction=None):
        visible = dict(observation)
        if visible.get("profile") == profile:
            visible.pop("profile")
        # read_reviews is a derived index of get_reviews results. Keep any
        # unmatched row, but do not send the same full review twice.
        returned = {}
        for item in visible.get("tool_results", []):
            if item.get("action", {}).get("tool_name") != "get_reviews":
                continue
            asin = item["action"].get("asin")
            for review in (item.get("result", {}).get("data") or {}).get("reviews", []):
                returned.setdefault(asin, []).append(review)
        if isinstance(visible.get("read_reviews"), dict):
            extra = {
                asin: [review for review in reviews if review not in returned.get(asin, [])]
                for asin, reviews in visible["read_reviews"].items()
            }
            extra = {asin: reviews for asin, reviews in extra.items() if reviews}
            if extra:
                visible["read_reviews"] = extra
            else:
                visible.pop("read_reviews")
        payload = {"agent_id": agent_id, "profile": profile, "observation": visible}
        if correction:
            payload["format_correction"] = correction
        return self.client.complete_json(REVIEW_EVENT_SYSTEM, payload)

    def decide_purchase(self, agent_id, profile, observation, correction=None):
        payload = {"agent_id": agent_id, "profile": profile, "observation": observation}
        if correction:
            payload["format_correction"] = correction
        return self.client.complete_json(PURCHASE_DECISION_SYSTEM, payload)

    def decide_simulation(self, agent_id, profile, observation, correction=None):
        """Compatibility alias; new callers should use decide_purchase."""
        return self.decide_purchase(agent_id, profile, observation, correction)
