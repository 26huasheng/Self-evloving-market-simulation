"""Immutable checkpoints and atomically committed run manifest."""

import json
import os
import re
import tempfile
from pathlib import Path
import torch
from sems_attention import NeuralAttentionPolicy
from .types import PolicyVersion, serialize


class History:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = self.directory / "history.json"
        self.data = (
            json.loads(self.manifest.read_text(encoding="utf-8"))
            if self.manifest.exists()
            else {"current": None, "versions": {}, "records": []}
        )

    @property
    def current(self):
        return PolicyVersion(**self.data["current"]) if self.data["current"] else None

    def _atomic(self, path, writer):
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="rsi-")
        os.close(fd)
        try:
            writer(tmp)
            # Windows' FlushFileBuffers requires a writable file handle.
            with open(tmp, "rb+") as f:
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _id(self, version):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", version) or version in (".", ".."):
            raise ValueError("invalid version ID")

    def save_policy(self, version, policy):
        self._id(version)
        path = self.directory / (version + ".pt")
        if path.exists():
            raise FileExistsError(path)
        self._atomic(path, policy.save)
        return str(path)

    def save_optimizer_state(self, version, state):
        self._id(version)
        path = self.directory / (version + ".optimizer.pt")
        if path.exists():
            raise FileExistsError(path)
        self._atomic(path, lambda p: torch.save(state, p))
        return str(path)

    def restore(self, version):
        entry = self.data["versions"][version]
        return NeuralAttentionPolicy.load(entry["checkpoint"])

    def _commit(self, data):
        encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        self._atomic(self.manifest, lambda p: Path(p).write_text(encoded, encoding="utf-8"))
        self.data = data

    def initialize(self, policy, version="initial", config=None):
        if self.current:
            raise ValueError("history already initialized")
        path = self.save_policy(version, policy)
        v = serialize(PolicyVersion(version, path, config or {}, True))
        self._commit({"current": v, "versions": {version: v}, "records": []})
        return self.current

    def record(self, record):
        import copy

        data = copy.deepcopy(self.data)
        candidate = record.candidate
        if candidate["version_id"] in data["versions"]:
            raise ValueError("version already recorded")
        v = serialize(
            PolicyVersion(
                candidate["version_id"],
                candidate["checkpoint"],
                candidate["config"],
                record.validation["status"] == "accepted",
            )
        )
        data["versions"][v["version_id"]] = v
        data["records"].append(serialize(record))
        if record.validation["status"] == "accepted":
            data["current"] = v
        self._commit(data)

    def rollback(self, version):
        import copy

        data = copy.deepcopy(self.data)
        target = data["versions"][version]
        if not target["verified"]:
            raise ValueError("cannot promote unverified version")
        self.restore(version)  # validate checkpoint before changing the pointer
        data.setdefault("rollbacks", []).append(
            {
                "from_version": data["current"]["version_id"],
                "to_version": version,
                "after_round": len(data["records"]),
            }
        )
        data["current"] = target
        self._commit(data)
