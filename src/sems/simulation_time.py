"""Case-relative clock and review-event-conditioned session schedule.

Review timestamps reconstruct observed review activity; they do not predict purchase time.
Ground truth is returned separately and must never be passed to agents or tools.
"""

from dataclasses import dataclass, asdict
import hashlib

DAY_MS = 86_400_000


@dataclass(frozen=True)
class SimulationDay:
    index: int
    start_ms: int
    end_ms: int


class Clock:
    def __init__(self, t0_ms, end_ms):
        if type(t0_ms) is not int or type(end_ms) is not int or end_ms <= t0_ms:
            raise ValueError("invalid case window")
        self.t0_ms, self.end_ms = t0_ms, end_ms

    def __len__(self):
        return (self.end_ms - self.t0_ms + DAY_MS - 1) // DAY_MS

    def day(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        start = self.t0_ms + index * DAY_MS
        return SimulationDay(index, start, min(start + DAY_MS, self.end_ms))

    def __iter__(self):
        return (self.day(i) for i in range(len(self)))


@dataclass(frozen=True)
class Session:
    session_id: str
    user_id: str
    scheduled_at_ms: int
    simulation_day: int

    def to_dict(self):
        return asdict(self)


def build_schedule(case_store):
    """Return agent-safe sessions and a separate session_id -> review mapping."""
    clock = Clock(case_store.t0_ms, case_store.end_ms)
    events = []
    for asin in case_store.asins:
        for row in case_store.reviews[asin]:
            timestamp = row.get("timestamp_ms")
            if (
                type(timestamp) is int
                and clock.t0_ms <= timestamp < clock.end_ms
                and row.get("user_id") in case_store.users
            ):
                if row.get("asin") != asin or not row.get("review_id"):
                    raise ValueError("invalid future review event")
                events.append(row)
    events.sort(key=lambda r: (r["timestamp_ms"], r["review_id"], r["user_id"], r["asin"]))
    sessions, truth = [], {}
    for index, row in enumerate(events):
        token = hashlib.sha256((str(index) + ":" + row["review_id"]).encode()).hexdigest()[:24]
        sid = "session_" + token
        sessions.append(
            Session(
                sid,
                row["user_id"],
                row["timestamp_ms"],
                (row["timestamp_ms"] - clock.t0_ms) // DAY_MS,
            )
        )
        truth[sid] = dict(row)
    return sessions, truth
