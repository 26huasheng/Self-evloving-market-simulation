"""Read validated four-stage profiles without generating missing profiles."""

import hashlib
import json
from pathlib import Path

from .four_stage import PROMPT_VERSION, validate_profile


class CachedProfiles:
    def __init__(self, directory, model="gpt-4.1-mini", chunk_size=20, client=None):
        self.directory = Path(directory)
        self.model = model
        self.chunk_size = chunk_size
        self.client = client
        self.hits = 0

    def load(self, user_id, history):
        digest = hashlib.sha256(
            json.dumps(history, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        version = hashlib.sha256(
            (self.model + PROMPT_VERSION + str(self.chunk_size)).encode()
        ).hexdigest()
        path = self.directory / f"{user_id}_{digest}_{version}.json"
        if not path.exists():
            raise FileNotFoundError("validated four-stage profile cache missing")
        profile = validate_profile(json.loads(path.read_text(encoding="utf-8")), history)
        self.hits += 1
        if self.client:
            self.client.record_cache_hit("Profile Extraction", self.model)
        return profile
