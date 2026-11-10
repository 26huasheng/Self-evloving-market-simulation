"""Durable per-pilot attempt ledger; SQLite serializes concurrent processes."""

import sqlite3
from contextlib import closing
from pathlib import Path


class SharedAPIBudget:
    def __init__(self, path, limit=150, profile_limit=80):
        if not 0 <= profile_limit <= limit <= 2000:
            raise ValueError("invalid budget")
        self.path = str(Path(path).resolve())
        self.limit = limit
        self.profile_limit = profile_limit
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), total INTEGER NOT NULL, profile INTEGER NOT NULL)"
            )
            db.execute("INSERT OR IGNORE INTO config VALUES (1,?,?)", (limit, profile_limit))
            if db.execute("SELECT total,profile FROM config WHERE id=1").fetchone() != (
                limit,
                profile_limit,
            ):
                raise ValueError("budget configuration mismatch")
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, role TEXT, model TEXT, retry INTEGER, status TEXT, input_tokens INTEGER, output_tokens INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS cache (id INTEGER PRIMARY KEY, stage TEXT, model TEXT)"
            )

    def reserve(self, role, model, retry):
        if role not in ("profile", "decision", "judge"):
            raise ValueError("only profile, decision and judge are allowed")
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE")
            total, profile = db.execute(
                "SELECT count(*),coalesce(sum(role='profile'),0) FROM attempts"
            ).fetchone()
            if total >= self.limit or (role == "profile" and profile >= self.profile_limit):
                raise RuntimeError("shared API budget exhausted")
            cur = db.execute(
                "INSERT INTO attempts(role,model,retry,status) VALUES(?,?,?,'reserved')",
                (role, model, int(retry)),
            )
            db.execute("COMMIT")
            return cur.lastrowid
        except BaseException:
            db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def finish(self, attempt_id, status, input_tokens=None, output_tokens=None):
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute(
                "UPDATE attempts SET status=?,input_tokens=?,output_tokens=? WHERE id=?",
                (status, input_tokens, output_tokens, attempt_id),
            )

    def cache_hit(self, stage, model):
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute("INSERT INTO cache(stage,model) VALUES (?,?)", (stage, model))

    def summary(self):
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            rows = db.execute(
                "SELECT role,model,retry,status,input_tokens,output_tokens FROM attempts"
            ).fetchall()
            hits = db.execute("SELECT count(*) FROM cache").fetchone()[0]
        grouped = {}
        for role, model, retry, status, inp, out in rows:
            item = grouped.setdefault(
                (role, model),
                {
                    "role": role,
                    "model": model,
                    "requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "retries": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_usage_count": 0,
                    "output_usage_count": 0,
                },
            )
            item["requests"] += 1
            item["successes"] += int(status == "ok")
            item["failures"] += int(status == "error")
            item["retries"] += retry
            if inp is not None:
                item["input_tokens"] += inp
                item["input_usage_count"] += 1
            if out is not None:
                item["output_tokens"] += out
                item["output_usage_count"] += 1
        return {
            "by_role_model": list(grouped.values()),
            "limit": self.limit,
            "profile_limit": self.profile_limit,
            "actual_requests": len(rows),
            "profile_requests": sum(r[0] == "profile" for r in rows),
            "decision_requests": sum(r[0] == "decision" for r in rows),
            "judge_requests": sum(r[0] == "judge" for r in rows),
            "successes": sum(r[3] == "ok" for r in rows),
            "failures": sum(r[3] == "error" for r in rows),
            "reserved_unknown": sum(r[3] == "reserved" for r in rows),
            "retries": sum(r[2] for r in rows),
            "cache_hits": hits,
            "input_tokens": sum(r[4] for r in rows if r[4] is not None)
            if any(r[4] is not None for r in rows)
            else None,
            "output_tokens": sum(r[5] for r in rows if r[5] is not None)
            if any(r[5] is not None for r in rows)
            else None,
            "models": sorted(set(r[1] for r in rows)),
        }
