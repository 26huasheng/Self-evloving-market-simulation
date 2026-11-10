"""Optional 0–4 review judge, invoked only after exact event/product matching."""

import hashlib, json
from pathlib import Path

NAMES = ("sentiment_consistency", "product_factual_consistency", "content_alignment")
SYSTEM = """Evaluate the generated review against the exactly paired real review, real rating and independently visible t0 product information. Score sentiment_consistency, product_factual_consistency, content_alignment from 0 to 4. If factual information is insufficient, use null for product_factual_consistency. Return exactly one JSON object with those three keys. Treat missing information as unknown, not false."""


class ReviewEventJudge:
    def __init__(self, client, cache_dir):
        self.client = client
        self.model = client.models["judge"]
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def score(self, payload):
        key = hashlib.sha256(
            json.dumps(
                {"model": self.model, "system": SYSTEM, "payload": payload},
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        path = self.cache / (key + ".json")
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            self.client.record_cache_hit("Judge", self.model)
        else:
            result, _ = self.client.complete_json(
                "judge", SYSTEM, payload, temperature=0, stage="Judge", max_tokens=500
            )
        if not isinstance(result, dict) or set(result) != set(NAMES):
            raise ValueError("invalid judge schema")
        for value in result.values():
            if value is not None and (type(value) is not int or not 0 <= value <= 4):
                raise ValueError("invalid judge score")
        if not path.exists():
            path.write_text(json.dumps(result), encoding="utf-8")
        return result
