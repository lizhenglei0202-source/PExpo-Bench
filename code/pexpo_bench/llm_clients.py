"""Model clients for the four evaluated models and the auxiliary grounding judge.

Requests, decoding settings and token accounting follow the experiment protocol.
The registered prices describe the manuscript cost calculation.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal

# Optional deps; importing lazily inside each backend
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore



# ==========================================================================
# Response container
# ==========================================================================
@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    backend: str
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


# ==========================================================================
# Model registry — snapshot IDs locked for reproducibility
# ==========================================================================
MODEL_REGISTRY: dict[str, dict] = {
    'gpt-5.4': {"backend": "openai_compat", "model": "gpt-5.4",
                      "base_url": (os.environ.get("OPENAI_BASE_URL_NATIVE") or "https://api.openai.com/v1"),
                      "_native_openai": True,
                      "price_in": 2.5, "price_out": 15.0},
    'gpt-5.4-nano': {"backend": "openai_compat", "model": "gpt-5.4-nano",
                      "base_url": (os.environ.get("OPENAI_BASE_URL_NATIVE") or "https://api.openai.com/v1"),
                      "price_in": 0.20, "price_out": 1.25,
                      "_native_openai": True},
    'gpt-5.4-mini': {"backend": "openai_compat", "model": "gpt-5.4-mini",
                      "base_url": (os.environ.get("OPENAI_BASE_URL_NATIVE") or "https://api.openai.com/v1"),
                      "price_in": 0.75, "price_out": 4.50,
                      "_native_openai": True},
    'gpt-4o-mini': {"backend": "openai_compat", "model": "gpt-4o-mini",
                      "base_url": (os.environ.get("OPENAI_BASE_URL_NATIVE") or "https://api.openai.com/v1"),
                      "_native_openai": True,
                      "price_in": 0.15, "price_out": 0.6},
    'deepseek-v4': {"backend": "openai_compat", "model": "deepseek-v4-flash",
                         "base_url": "https://api.deepseek.com",
                         "price_in": 0.14, "price_out": 0.28},
}


# ==========================================================================
# Unified client
# ==========================================================================
class LLMClient:
    """Call any registered model with a unified interface."""

    def __init__(self, model_key: str, temperature: float = 0.3,
                 max_tokens: int = 2048, seed: int | None = 42):
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model_key: {model_key}")
        self.cfg = MODEL_REGISTRY[model_key]
        self.model_key = model_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self._init_backend()

    # ---------------------- backend init ----------------------
    def _init_backend(self) -> None:
        backend = self.cfg["backend"]
        if backend in ("openai", "openai_compat"):
            if OpenAI is None:
                raise ImportError("pip install openai")
            kwargs = {}
            if backend == "openai_compat":
                kwargs["base_url"] = self.cfg["base_url"]
                kwargs["api_key"] = os.environ.get(
                    self._env_key(), "EMPTY"
                )
            else:
                kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
            self._client = OpenAI(timeout=60.0, max_retries=3, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _env_key(self) -> str:
        base = self.cfg.get("base_url", "")
        # Native OpenAI uses a separate project-scoped key
        if self.cfg.get("_native_openai") or base == "https://api.openai.com/v1":
            return "OPENAI_API_KEY_NATIVE"
        if "deepseek" in base:
            return "deepseek_API_KEY"
        return "OPENAI_API_KEY"

    # ---------------------- main call ----------------------
    def chat(self, messages: list[dict],
             temperature: float | None = None,
             max_tokens: int | None = None) -> LLMResponse:
        backend = self.cfg["backend"]
        t0 = time.time()
        if backend in ("openai", "openai_compat"):
            # OpenAI native API for gpt-5.x deprecated max_tokens → use max_completion_tokens
            tokens_kwarg = ('max_completion_tokens'
                            if self.cfg.get("_native_openai")
                            else 'max_tokens')
            # The completion budget covers reasoning and visible output.
            # Use the protocol's 16,384-token floor for these model endpoints.
            tok = max_tokens or self.max_tokens
            if self.cfg.get("_native_openai"):
                tok = max(tok, 16384)
            call_kwargs = {
                'model': self.cfg["model"],
                'messages': messages,
                'temperature': temperature if temperature is not None else self.temperature,
                tokens_kwarg: tok,
            }
            if self.seed is not None and not self.cfg.get("_native_openai"):
                call_kwargs['seed'] = self.seed
            try:
                resp = self._client.chat.completions.create(**call_kwargs)
                content = resp.choices[0].message.content or ""
                in_tok = resp.usage.prompt_tokens
                out_tok = resp.usage.completion_tokens
                finish = resp.choices[0].finish_reason
            except Exception as e:
                # Auth / permission errors mean the whole run is misconfigured — fail loudly
                # instead of treating missing responses as completed predictions.
                name = type(e).__name__
                if "Authentication" in name or "PermissionDenied" in name or getattr(e, "status_code", None) in (401, 402, 403):
                    raise
                # A quota-exhausted 429 is a dead account, not a transient rate
                # limit; fail immediately when provider credits are exhausted.
                if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                    raise
                # content-filter / invalid_prompt / transient API error: return an
                # empty response so the item is recorded as a failure (parse_error)
                # rather than crashing the whole run.
                content, in_tok, out_tok, finish = "", 0, 0, f"error:{name}"
        else:
            raise ValueError(backend)

        return LLMResponse(
            content=content, input_tokens=in_tok, output_tokens=out_tok,
            latency_s=time.time() - t0, model=self.cfg["model"],
            backend=backend, finish_reason=finish,
        )

    # ---------------------- cost helper ----------------------
    def cost_usd(self, r: LLMResponse) -> float:
        return (r.input_tokens * self.cfg["price_in"]
                + r.output_tokens * self.cfg["price_out"]) / 1_000_000


# ==========================================================================
# Module-level convenience
# ==========================================================================
_default_client: LLMClient | None = None


def get_client(model_key: str = "gpt-5.4") -> LLMClient:
    global _default_client
    if _default_client is None or _default_client.model_key != model_key:
        _default_client = LLMClient(model_key)
    return _default_client
