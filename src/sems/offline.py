"""Explicit offline smoke-test doubles. No API or learned behavior is involved."""

import hashlib

import numpy as np


class FixtureEmbeddings:
    """Deterministic local vectors, for plumbing checks only."""

    def __init__(self):
        self.cache = {}

    def encode(self, text):
        if text not in self.cache:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
            vector = np.random.default_rng(seed).standard_normal(384).astype("float32")
            self.cache[text] = vector / np.linalg.norm(vector)
        return self.cache[text]

    def mean(self, texts):
        values = [self.encode(text) for text in texts if text]
        return np.mean(values, axis=0) if values else np.zeros(384, dtype="float32")


class OfflineProfiles:
    def load(self, user_id, history):
        return {
            "user_id": user_id,
            "narrative": "Offline integration fixture; no inferred consumer preferences.",
            "preferences": [],
            "uncertainties": ["Test fixture, not a formal consumer profile."],
            "profile_type": "offline_fixture",
        }


class OfflineDecision:
    def decide_review_event(self, agent_id, profile, observation, correction=None):
        return {
            "selected_asin": observation["shelf"][0]["product_id"],
            "rating_probabilities": [0.0, 0.0, 0.0, 1.0, 0.0],
            "rating": 4,
            "review_text": "Offline simulation test review.",
        }

    def decide_simulation(self, agent_id, profile, observation, correction=None):
        return {
            "purchase": True,
            "product_id": observation["shelf"][0]["product_id"],
            "submit_rating": True,
            "rating": 4,
            "publish_review": True,
            "review_text": "Offline simulation test review.",
            "rationale": "Offline test decision.",
        }

    def decide_purchase(self, agent_id, profile, observation, correction=None):
        # Dispatch through the old method so older test subclasses still work.
        return self.decide_simulation(agent_id, profile, observation, correction)
