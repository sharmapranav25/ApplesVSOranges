"""Inference backends for joke explanation generation.

Reuses the chat-completion backends from `judge_backend.py` (Ollama, OpenRouter,
Together) for any OpenAI-compatible endpoint, and adds a `GeminiBackend` for
Google AI Studio's free tier (Gemini's API is not OpenAI-compatible).

For local Qwen / R1 / Llama runs, use the Ollama backend with the appropriate
model tag — `OllamaBackend(model="deepseek-r1:7b")`. The Ollama daemon you
already have running for the judge serves these the same way.
"""

from __future__ import annotations

import os
from typing import Protocol

import requests

# Re-export the chat-completion backends so callers don't need to know which
# module they live in.
from judge_backend import (
    OllamaBackend,
    OpenRouterBackend,
    TogetherBackend,
)


class InferenceBackend(Protocol):
    name: str
    def complete(self, messages: list[dict], temperature: float, max_tokens: int) -> str: ...


# ---------- Gemini (Google AI Studio free tier) ----------

class GeminiBackend:
    """Gemini 1.5 Flash / Pro via Google AI Studio's free tier.

    Get a free API key at https://aistudio.google.com/apikey (no card).

    Free-tier rate limits (verify current values; they shift):
        gemini-1.5-flash : ~15 req/min, ~1500 req/day
        gemini-1.5-pro   : ~ 2 req/min, ~  50 req/day  (slow — for 600 jokes
                                                       you'll need --sleep 31)

    For a 600-joke run, Flash is comfortable (~40 min at 15 rpm).
    Pro at 50/day means ~12 days unless you upgrade — plan accordingly.
    """
    name = "gemini"
    DEFAULT_MODEL = "gemini-1.5-flash"
    URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 timeout: float = 90.0):
        self.model = model or os.environ.get("GEMINI_MODEL", self.DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get one free at "
                "https://aistudio.google.com/apikey and `export GEMINI_API_KEY=...`"
            )
        self.timeout = timeout

    def complete(self, messages: list[dict], temperature: float,
                 max_tokens: int = 300) -> str:
        # Gemini doesn't use OpenAI-style multi-turn messages. Our use case is
        # single-turn (one user prompt → one explanation), so we flatten.
        prompt = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")

        url = self.URL_TMPL.format(model=self.model)
        resp = requests.post(
            url,
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
                # Some jokes in the dataset trip Gemini's default safety
                # filters (paper notes the same problem). Loosen to BLOCK_NONE
                # so we can replicate the paper's coverage; the writeup should
                # mention this.
                "safetySettings": [
                    {"category": cat, "threshold": "BLOCK_NONE"}
                    for cat in (
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                    )
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Defensive parse — Gemini may return no candidates if a request is
        # blocked even with BLOCK_NONE (rare but happens).
        candidates = data.get("candidates", [])
        if not candidates:
            block = data.get("promptFeedback", {})
            raise RuntimeError(f"Gemini returned no candidates (blocked?): {block}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            finish = candidates[0].get("finishReason", "?")
            raise RuntimeError(f"Gemini returned no parts (finishReason={finish})")
        return parts[0].get("text", "")


def get_inference_backend(name: str, model: str | None = None) -> InferenceBackend:
    """Factory: returns a backend instance for joke-explanation inference.

    `model` overrides the backend's default model id (Ollama tag, Gemini model
    name, etc.). For OpenRouter/Together the model is fixed in the class — pass
    a different backend if you want a different model on those providers.
    """
    if name == "ollama":
        return OllamaBackend(model=model)
    if name == "gemini":
        return GeminiBackend(model=model)
    if name == "openrouter":
        return OpenRouterBackend()
    if name == "together":
        return TogetherBackend()
    raise ValueError(f"unknown backend: {name!r}")
