"""OpenAI client wrapper — mirror of src/client.py for the multi-model comparison.

Uses the Responses API so reasoning models return reasoning summaries (the
closest available analog to Claude's thinking traces — OpenAI does not expose
raw reasoning tokens). Shares the same Redis cache; the model name is part of
every cache key.

Two modes, mirroring call_claude:
- reasoning=False: plain completion (knowledge gate, judge-style calls).
- reasoning=True:  reasoning model with effort + summary capture.
"""

import os

from openai import OpenAI

from src.cache import cache_get, cache_set, make_key

# Tier-matched to claude-sonnet-4-6 (comparable per-token cost) so cross-model
# differences can't be attributed to capability tier. Note gpt-5.4 is absent
# from models.list() but is callable directly.
#
# Reasoning-token scale differs sharply from Claude's thinking-character scale
# (median ~49 tokens at high effort vs Claude's ~300 chars), so thin/thick
# thresholds must be derived per-model from each one's own distribution rather
# than shared across families.
DEFAULT_REASONING_MODEL = "gpt-5.4"
DEFAULT_PLAIN_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "high"  # parity with Claude's extended thinking

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def call_openai(
    prompt: str,
    *,
    cache_key_parts: list[str],
    model: str | None = None,
    reasoning: bool = False,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = 4000,
) -> dict:
    """Call an OpenAI model with caching.

    Returns dict with keys: answer, reasoning_summary, usage, cached.
    reasoning_summary is "" when reasoning=False or no summary is returned.
    """
    key = make_key(*cache_key_parts)
    cached = cache_get(key)
    if cached is not None:
        return {
            "answer": cached["answer"],
            "reasoning_summary": cached.get("reasoning_summary", ""),
            "usage": cached.get("usage"),
            "cached": True,
        }

    client = get_client()
    model = model or (DEFAULT_REASONING_MODEL if reasoning else DEFAULT_PLAIN_MODEL)

    kwargs = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning:
        kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}

    response = client.responses.create(**kwargs)

    answer = response.output_text or ""
    summary_parts = []
    for item in response.output:
        if getattr(item, "type", None) == "reasoning":
            for s in getattr(item, "summary", []) or []:
                text = getattr(s, "text", "")
                if text:
                    summary_parts.append(text)
    reasoning_summary = "\n".join(summary_parts)

    usage = None
    if getattr(response, "usage", None) is not None:
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        details = getattr(response.usage, "output_tokens_details", None)
        if details is not None:
            usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0)

    payload = {"answer": answer, "reasoning_summary": reasoning_summary, "usage": usage}
    cache_set(key, payload)
    return {**payload, "cached": False}
