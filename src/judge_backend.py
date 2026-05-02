"""Switchable Qwen2.5-7B-Instruct backends for the judge.

Four implementations:
- OpenRouterBackend: cheap+fast remote inference. Needs $OPENROUTER_API_KEY.
- TogetherBackend: alternative remote. Needs $TOGETHER_API_KEY.
- OllamaBackend (recommended free, local): runs against a local `ollama serve`
  instance. No key, no rate limits. ~3-5 hrs for the full 9600-call run on Mac+MPS
  with the q4_K_M quant.
- LocalBackend: HF transformers + bitsandbytes 4-bit. ~5GB VRAM at int4 / ~16GB at bf16.
  Still bitsandbytes-blocked on macOS (no Darwin wheels); use Ollama instead on Mac.

NOTE: paper uses Qwen2.5-72B-Instruct; we deviate to 7B for cost/feasibility (Member 1
decision, 2026-04-30). All slugs below switched accordingly.

All four expose `.complete(messages: list[dict], temperature: float) -> str`.
"""

from __future__ import annotations

import os
from typing import Protocol

import requests


class JudgeBackend(Protocol):
    name: str
    def complete(self, messages: list[dict], temperature: float, max_tokens: int = 16) -> str: ...


# ---------- OpenRouter ----------

class OpenRouterBackend:
    name = "openrouter"
    # Qwen2.5-7B-Instruct on OpenRouter — verify the exact slug at:
    # https://openrouter.ai/models?q=qwen2.5-7b-instruct
    MODEL_ID = "qwen/qwen-2.5-7b-instruct"
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.timeout = timeout

    def complete(self, messages: list[dict], temperature: float, max_tokens: int = 16) -> str:
        resp = requests.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.MODEL_ID,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------- Together ----------

class TogetherBackend:
    name = "together"
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-Turbo"  # Together's served variant
    URL = "https://api.together.xyz/v1/chat/completions"

    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise RuntimeError("TOGETHER_API_KEY not set")
        self.timeout = timeout

    def complete(self, messages: list[dict], temperature: float, max_tokens: int = 16) -> str:
        resp = requests.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.MODEL_ID,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------- Ollama (local, free) ----------

class OllamaBackend:
    """Local Qwen2.5-7B-Instruct via Ollama's OpenAI-compatible HTTP API.

    Setup:
        brew install ollama
        ollama serve &                          # background daemon on :11434
        ollama pull qwen2.5:7b-instruct-q4_K_M  # ~4.4 GB

    Override defaults via env: OLLAMA_URL, OLLAMA_MODEL.
    No API key required; no rate limits.
    """
    name = "ollama"
    DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
    DEFAULT_URL = "http://localhost:11434/v1/chat/completions"

    def __init__(self, model: str | None = None, url: str | None = None,
                 timeout: float = 180.0):
        self.model = model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.url = url or os.environ.get("OLLAMA_URL", self.DEFAULT_URL)
        # Local inference is slower per call than a hosted API — bigger timeout.
        self.timeout = timeout

    def complete(self, messages: list[dict], temperature: float, max_tokens: int = 16) -> str:
        resp = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------- Local (4-bit) ----------

class LocalBackend:
    """Local Qwen2.5-7B-Instruct via HF transformers + bitsandbytes 4-bit.

    Heavy import; constructed lazily so the module doesn't pay the cost on
    --backend openrouter runs.
    """
    name = "local"
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

    def complete(self, messages: list[dict], temperature: float, max_tokens: int = 16) -> str:
        torch = self._torch
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text


def get_backend(name: str) -> JudgeBackend:
    if name == "openrouter":
        return OpenRouterBackend()
    if name == "together":
        return TogetherBackend()
    if name == "ollama":
        return OllamaBackend()
    if name == "local":
        return LocalBackend()
    raise ValueError(f"unknown backend: {name!r}")
