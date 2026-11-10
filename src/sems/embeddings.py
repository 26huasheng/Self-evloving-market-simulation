"""Frozen MiniLM embeddings with full-text token-aware chunking."""

import hashlib
import os
from pathlib import Path

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODING_VERSION = "token_chunks_weighted_v1"


class TextEmbeddings:
    def __init__(self, model=None, *, model_path=None, local_files_only=False):
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for MiniLM embeddings"
                ) from exc
            source = model_path or os.getenv("SEMS_EMBEDDING_MODEL_PATH") or MODEL_ID
            self.model_source = str(source)
            model = SentenceTransformer(
                str(source), local_files_only=local_files_only or Path(source).is_dir()
            )
        else:
            self.model_source = "injected_model"
        self.model = model
        self.tokenizer = model.tokenizer
        self.content_limit = int(model.max_seq_length) - self.tokenizer.num_special_tokens_to_add(
            pair=False
        )
        if self.content_limit <= 0:
            raise ValueError("MiniLM sequence limit leaves no room for content")
        self.cache = {}

    def chunks(self, text):
        """Partition every source character at tokenizer boundaries, without truncation."""
        if not isinstance(text, str):
            raise TypeError("embedding text must be a string")
        offsets = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )["offset_mapping"]
        if len(offsets) <= self.content_limit:
            return [(text, len(offsets))]
        chunks = []
        start_char = 0
        token_index = 0
        while token_index < len(offsets):
            end_index = min(token_index + self.content_limit, len(offsets))
            while True:
                end_char = offsets[end_index - 1][1] if end_index < len(offsets) else len(text)
                chunk = text[start_char:end_char]
                count = len(
                    self.tokenizer.encode(
                        chunk, add_special_tokens=False, truncation=False, verbose=False
                    )
                )
                if count <= self.content_limit:
                    break
                end_index -= 1
                if end_index <= token_index:
                    raise ValueError("one tokenizer span exceeds the MiniLM content limit")
            chunks.append((chunk, count))
            start_char = end_char
            token_index = end_index
        if "".join(chunk for chunk, _ in chunks) != text:
            raise RuntimeError("embedding chunking lost source text")
        return chunks

    def encode(self, text):
        key = (MODEL_ID, ENCODING_VERSION, hashlib.sha256(text.encode()).hexdigest())
        if key not in self.cache:
            chunks = self.chunks(text)
            if len(chunks) == 1:
                value = np.asarray(self.model.encode(text), dtype="float32")
            else:
                vectors = np.asarray(
                    self.model.encode([chunk for chunk, _ in chunks]), dtype="float32"
                )
                weights = np.asarray([count for _, count in chunks], dtype="float32")
                value = np.average(vectors, axis=0, weights=weights).astype("float32")
                norm = float(np.linalg.norm(value))
                if norm > 0:
                    value /= norm
            if value.shape != (384,) or not np.isfinite(value).all():
                raise ValueError("embedding must be a finite 384-dimensional vector")
            self.cache[key] = value
        return self.cache[key]

    def mean(self, texts):
        values = [self.encode(t) for t in texts if t]
        return np.mean(values, axis=0) if values else np.zeros(384, dtype="float32")
