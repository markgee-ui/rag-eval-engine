"""Thin wrapper around the Claude API: one function that returns validated JSON."""

import json
import os

import anthropic

MODEL = os.environ.get("RAG_MODEL", "claude-opus-5-5")

_client: anthropic.Anthropic | None = None


class LLMError(RuntimeError):
    pass


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def structured_call(system: str, prompt: str, schema: dict, *, model: str = MODEL,
                    effort: str = "medium", max_tokens: int = 8000) -> dict:
    """Call Claude and get back a dict that matches `schema`.

    Structured outputs (output_config.format) guarantee valid JSON, so there is
    no regex parsing of free text. `fallbacks="default"` lets the API retry on a
    fallback model if a safety classifier declines the request (legal text
    about breaches/indemnities can occasionally trip one).
    """
    response = client().beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": schema},
        },
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise LLMError(f"Model declined the request (category={category})")
    if response.stop_reason == "max_tokens":
        raise LLMError("Response hit max_tokens before the JSON was complete")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise LLMError(f"No text block in response (stop_reason={response.stop_reason})")
    return json.loads(text)
