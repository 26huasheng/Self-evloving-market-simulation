"""Shared OpenAI SDK client with strict call caps and redacted usage logs."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    elapsed_seconds: float
    attempts: int
    estimated_cost: float | None


class SEMSAPIClient:
    """OpenAI-compatible Chat Completions, one shared client for SEMS modules.

    Cost is only calculated when a current rate snapshot is explicitly supplied;
    this avoids silently applying stale first-party or third-party prices.
    """

    def __init__(
        self,
        *,
        env_file: str | Path | None = None,
        max_requests: int | None = None,
        max_retries: int | None = None,
        usage_log: str | Path | None = None,
        prices: Mapping[str, Mapping[str, float]] | None = None,
        attempt_budget=None,
    ):
        default_env = os.getenv("SEMS_ENV_FILE") or ROOT / ".env"
        load_dotenv(dotenv_path=env_file or default_env, override=False)
        api_key = os.getenv("SEMS_API_KEY")
        if not api_key:
            raise RuntimeError("SEMS_API_KEY is missing; set it in the root .env file")
        self.base_url = os.getenv("SEMS_BASE_URL", "https://api.ai-gaochao.cn/v1").rstrip("/")
        self.models = {
            "profile": os.getenv("SEMS_PROFILE_MODEL", "gpt-4o-mini"),
            "decision": os.getenv("SEMS_DECISION_MODEL", "gpt-4o-mini"),
            "judge": os.getenv("SEMS_JUDGE_MODEL", "gpt-4.1-mini"),
            "proposer": os.getenv("SEMS_PROPOSER_MODEL", "gpt-4.1-mini"),
        }
        self.timeout = float(os.getenv("SEMS_API_TIMEOUT", "60"))
        self.max_requests = (
            max_requests
            if max_requests is not None
            else int(os.getenv("SEMS_API_MAX_REQUESTS", "2"))
        )
        self.max_retries = (
            max_retries if max_retries is not None else int(os.getenv("SEMS_API_MAX_RETRIES", "2"))
        )
        self.usage_log = (
            Path(usage_log)
            if usage_log
            else Path(os.getenv("SEMS_RUN_DIR", str(Path.home() / "SEMS-runtime/runs")))
            / "api_usage.jsonl"
        )
        self.prices = dict(prices or {})
        self.attempt_budget = attempt_budget
        # SDK retries are disabled; bounded retries below are counted explicitly.
        self._client = OpenAI(
            api_key=api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0
        )
        self._lock = threading.Lock()
        self.request_count = 0
        self.cumulative_tokens = 0
        self.cumulative_cost = 0.0
        self.cost_known = True
        self._attempt_events: list[dict[str, Any]] = []
        self._cache_hits: list[dict[str, str]] = []

    def record_cache_hit(self, stage: str, model: str):
        with self._lock:
            self._cache_hits.append({"stage": stage, "model": model})
        if self.attempt_budget:
            self.attempt_budget.cache_hit(stage, model)

    @staticmethod
    def _retryable(exc):
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in (408, 409, 429) or exc.status_code >= 500
        return False

    @staticmethod
    def _cost(model: str, prompt_tokens: int, completion_tokens: int, prices):
        rate = prices.get(model)
        if not rate:
            return None
        return (
            prompt_tokens * float(rate["input_per_million"])
            + completion_tokens * float(rate["output_per_million"])
        ) / 1_000_000

    def chat(
        self,
        *,
        module: str,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        agent_id: str | None = None,
        evolution_round: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 300,
        timeout: float | None = None,
        max_retries: int | None = None,
        stage: str | None = None,
    ) -> ChatResult:
        if module not in self.models:
            raise ValueError(f"Unknown module role: {module}")
        model_id = model or self.models[module]
        stage = (
            stage
            or {
                "profile": "Profile Extraction",
                "decision": "Decision",
                "judge": "Judge",
                "proposer": "Proposer",
            }[module]
        )
        retry_limit = self.max_retries if max_retries is None else max_retries
        with self._lock:
            if self.request_count >= self.max_requests:
                raise RuntimeError(
                    f"API request cap reached ({self.max_requests}); raise SEMS_API_MAX_REQUESTS deliberately to continue"
                )
            # Reserve a slot per actual attempt, not once per logical call.
            self.request_count += 1
        started = time.monotonic()
        prompt_bytes = json.dumps(list(messages), ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        last_exc = None
        result = None
        attempts = 0
        for attempt in range(retry_limit + 1):
            if attempt:
                with self._lock:
                    if self.request_count >= self.max_requests:
                        last_exc = RuntimeError("API request cap reached before retry")
                        break
                    self.request_count += 1
            attempt_id = (
                self.attempt_budget.reserve(module, model_id, bool(attempt))
                if self.attempt_budget
                else None
            )
            attempts += 1
            try:
                result = self._client.chat.completions.create(
                    model=model_id,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout or self.timeout,
                )
                usage = result.usage
                if self.attempt_budget:
                    self.attempt_budget.finish(
                        attempt_id,
                        "ok",
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                    )
                with self._lock:
                    self._attempt_events.append(
                        {
                            "stage": stage,
                            "model": getattr(result, "model", None) or model_id,
                            "requested_model": model_id,
                            "status": "ok",
                            "retry": bool(attempt),
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                        }
                    )
                last_exc = None
                break
            except Exception as exc:
                if self.attempt_budget:
                    self.attempt_budget.finish(attempt_id, "error")
                with self._lock:
                    self._attempt_events.append(
                        {
                            "stage": stage,
                            "model": model_id,
                            "status": "error",
                            "retry": bool(attempt),
                            "error_type": type(exc).__name__,
                            "prompt_tokens": None,
                            "completion_tokens": None,
                        }
                    )
                last_exc = exc
                if attempt >= retry_limit or not self._retryable(exc):
                    break
                retry_after = None
                if isinstance(exc, APIStatusError):
                    try:
                        retry_after = float(exc.response.headers.get("retry-after", ""))
                    except (ValueError, TypeError):
                        pass
                time.sleep(
                    min(30.0, max(1.0, retry_after if retry_after is not None else 2.0**attempt))
                )
        elapsed = time.monotonic() - started
        if result is None:
            self._log(
                {
                    "module": module,
                    "stage": stage,
                    "agent_id": agent_id,
                    "evolution_round": evolution_round,
                    "model": model_id,
                    "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                    "prompt_bytes": len(prompt_bytes),
                    "elapsed_seconds": round(elapsed, 4),
                    "attempts": attempts,
                    "status": "error",
                    "error_type": type(last_exc).__name__ if last_exc else "UnknownError",
                    "http_status": getattr(last_exc, "status_code", None),
                }
            )
            raise RuntimeError(
                f"SEMS API call failed for module={module}, model={model_id}; see redacted usage log"
            ) from last_exc
        usage = result.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        actual_model = getattr(result, "model", None) or model_id
        content = result.choices[0].message.content or ""
        cost = (
            self._cost(model_id, prompt_tokens, completion_tokens, self.prices)
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
        self.cumulative_tokens += total_tokens or 0
        if cost is None:
            self.cost_known = False
        else:
            self.cumulative_cost += cost
        self._log(
            {
                "module": module,
                "stage": stage,
                "agent_id": agent_id,
                "evolution_round": evolution_round,
                "model": model_id,
                "response_model": actual_model,
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "prompt_bytes": len(prompt_bytes),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cumulative_tokens": self.cumulative_tokens,
                "elapsed_seconds": round(elapsed, 4),
                "attempts": attempts,
                "status": "ok",
                "estimated_cost": cost,
                "cumulative_estimated_cost": self.cumulative_cost if self.cost_known else None,
                "cost_rate_source": "configured_current_rate" if cost is not None else None,
            }
        )
        return ChatResult(
            content,
            actual_model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            elapsed,
            attempts,
            cost,
        )

    def complete_json(
        self,
        module: str,
        system: str,
        payload: Mapping[str, Any],
        *,
        agent_id: str | None = None,
        evolution_round: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stage: str | None = None,
    ):
        # JSON is requested in the prompt, avoiding dependence on provider-side
        # response_format or native tool-call compatibility.
        result = self.chat(
            module=module,
            agent_id=agent_id,
            evolution_round=evolution_round,
            temperature=temperature,
            max_tokens=max_tokens,
            stage=stage,
            messages=[
                {"role": "system", "content": system + " Return valid JSON only."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        content = result.content.strip()
        # Some compatible providers wrap JSON in a Markdown fence despite the prompt.
        if content.startswith("```") and content.endswith("```"):
            first, separator, body = content.partition("\n")
            if separator and first.strip().lower() in ("```", "```json"):
                content = body[:-3].strip()
        try:
            return json.loads(content), result
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model returned invalid JSON for module={module} (length={len(content)}, position={exc.pos}, fenced={result.content.lstrip().startswith(chr(96) * 3)}); see redacted usage log"
            ) from exc

    def _log(self, record):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def usage_summary(self):
        stages = {}
        for event in self._attempt_events:
            key = (event["stage"], event["model"])
            row = stages.setdefault(
                key,
                {
                    "stage": key[0],
                    "model": key[1],
                    "actual_requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "retries": 0,
                    "cache_hits": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_usage_provided": 0,
                    "output_usage_provided": 0,
                },
            )
            row["actual_requests"] += 1
            row["successes" if event["status"] == "ok" else "failures"] += 1
            row["retries"] += int(event["retry"])
            for field, target, flag in (
                ("prompt_tokens", "input_tokens", "input_usage_provided"),
                ("completion_tokens", "output_tokens", "output_usage_provided"),
            ):
                if event[field] is not None:
                    row[target] += int(event[field])
                    row[flag] += 1
        for hit in self._cache_hits:
            key = (hit["stage"], hit["model"])
            row = stages.setdefault(
                key,
                {
                    "stage": key[0],
                    "model": key[1],
                    "actual_requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "retries": 0,
                    "cache_hits": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_usage_provided": 0,
                    "output_usage_provided": 0,
                },
            )
            row["cache_hits"] += 1
        rows = list(stages.values())
        for row in rows:
            row["input_usage_complete"] = row["input_usage_provided"] == row["actual_requests"]
            row["output_usage_complete"] = row["output_usage_provided"] == row["actual_requests"]
            if not row["input_usage_provided"]:
                row["input_tokens"] = None
            if not row["output_usage_provided"]:
                row["output_tokens"] = None
        return {
            "actual_requests": self.request_count,
            "successes": sum(x["successes"] for x in rows),
            "failures": sum(x["failures"] for x in rows),
            "retries": sum(x["retries"] for x in rows),
            "cache_hits": len(self._cache_hits),
            "stages": rows,
            "cumulative_tokens": self.cumulative_tokens,
            "cumulative_estimated_cost": self.cumulative_cost if self.cost_known else None,
            "cost_known": self.cost_known,
            "usage_log": str(self.usage_log),
        }

    def report_usage(self, output_path: str | Path | None = None):
        summary = self.usage_summary()
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("API_USAGE", json.dumps(summary, ensure_ascii=False))
        return summary
