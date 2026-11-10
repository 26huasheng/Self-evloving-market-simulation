"""Read-only loader for a self-contained SEMS case ZIP or directory."""

import json
import zipfile
from pathlib import Path
from datetime import datetime, timezone


class CaseStore:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(path) if self.path.is_file() else None
        names = (
            self.archive.namelist()
            if self.archive
            else [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]
        )
        self._names = set(names)
        if len(names) != len(self._names):
            raise ValueError("Duplicate archive paths")
        cases = [n for n in names if n == "case.json" or n.endswith("/case.json")]
        if len(cases) != 1:
            raise ValueError("Expected exactly one case.json")
        self.prefix = cases[0].rsplit("/", 1)[0] + "/" if "/" in cases[0] else ""
        self.case = self.json(cases[0])
        self.case_id = self.case["case_id"]
        self.t0_ms = self.case["t0_timestamp_ms"]
        self.end_ms = self.case["review_window_end_timestamp_ms"]
        if (
            not isinstance(self.t0_ms, int)
            or not isinstance(self.end_ms, int)
            or self.end_ms <= self.t0_ms
        ):
            raise ValueError("invalid case window")
        self.t0 = datetime.fromtimestamp(self.t0_ms / 1000, timezone.utc)
        self.end = datetime.fromtimestamp(self.end_ms / 1000, timezone.utc)
        self.shelf = self.jsonl(self.prefix + "shelf.jsonl")
        self.asins = tuple(x["product_id"] for x in self.shelf)
        if len(set(self.asins)) != len(self.asins):
            raise ValueError("Duplicate ASIN")
        self.products = {}
        self.series = {}
        self.reviews = {}
        self.users = {}
        for item in self.shelf:
            asin = item["product_id"]
            self.products[asin] = self.json(self.resolve(item["product_record_path"]))
            self.series[asin] = self.json(self.resolve(item["smoothed_history_path"]))["series"]
            self.reviews[asin] = self.jsonl(self.resolve(item["product_reviews_path"]))
        for name in names:
            if name.startswith(self.prefix + "user_profiles/") and name.endswith(".jsonl"):
                self.users[Path(name).stem] = self.jsonl(name)

    def resolve(self, name):
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or ".." in Path(name).parts
        ):
            raise ValueError("Unsafe path")
        candidates = list(dict.fromkeys((name, self.prefix + name)))
        found = [candidate for candidate in candidates if candidate in self._names]
        if len(found) != 1:
            raise ValueError(
                f"Case path {name!r} is {'ambiguous' if found else 'missing'}: {found}"
            )
        return found[0]

    def read(self, name):
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("Unsafe path")
        return self.archive.read(name) if self.archive else (self.path / name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def jsonl(self, name):
        return [
            json.loads(x) for x in self.read(name).decode("utf-8-sig").splitlines() if x.strip()
        ]

    def close(self):
        if self.archive:
            self.archive.close()
