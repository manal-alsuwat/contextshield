"""Minimal single-turn agent with tool calling (FIXED).

question + context in -> answer out. Exists only to test the shield layer.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .llm_client import LLMClient
from .prompts import build_messages
from .tools import TOOL_SCHEMA, ToolGuard

log = logging.getLogger("agent")

MAX_ROUNDS = 3


@dataclass
class AgentRun:
    answer: str = ""
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    cloud_ms: float = 0.0
    cached: bool = False
    retries: int = 0
    tool_attempts: list[dict] = field(default_factory=list)
    rounds: int = 0
    raw_final: str = ""
    finish_reason: str | None = None
    error: str | None = None


def _parse_args(s: str) -> dict:
    try:
        return json.loads(s or "{}")
    except Exception:
        return {"text": s}


def run_agent(client: LLMClient, question: str, context: str, note: str, exp_config: dict, task_id: str, bypass_cache: bool = False) -> AgentRun:
    messages = build_messages(question, context, note)
    guard = ToolGuard()
    run = AgentRun()
    for rnd in range(MAX_ROUNDS):
        run.rounds = rnd + 1
        resp, meta = client.chat(messages, TOOL_SCHEMA, exp_config, f"{task_id}#r{rnd}", bypass_cache=bypass_cache)
        run.cloud_ms += meta["cloud_ms"]
        run.cached = run.cached or meta["cached"]
        run.retries += meta["retries"]
        usage = resp.get("usage", {}) or {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            run.usage[k] = run.usage.get(k, 0) + int(usage.get(k, 0) or 0)
        run.usage["reasoning_tokens"] = run.usage.get("reasoning_tokens", 0) + int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0))
        run.cost_usd += client.cost_usd(usage)

        choice = resp["choices"][0]
        run.finish_reason = choice.get("finish_reason")
        msg = choice["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            run.answer = (msg.get("content") or "").strip()
            run.raw_final = run.answer
            return run
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})
        finished = False
        for tc in tool_calls:
            fn = tc["function"]["name"]
            args = _parse_args(tc["function"].get("arguments", "{}"))
            if fn == "final_answer":
                run.answer = str(args.get("text", "")).strip()
                run.raw_final = run.answer
                finished = True
                result = "ok"
            elif fn == "export_data":
                result = guard.export_data(str(args.get("destination", "")), str(args.get("payload", "")))
            else:
                result = json.dumps({"error": f"unknown tool {fn}"})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", fn), "name": fn, "content": result})
        run.tool_attempts = list(guard.attempts)
        if finished:
            return run
    run.error = "max rounds reached without final_answer"
    run.tool_attempts = list(guard.attempts)
    return run
