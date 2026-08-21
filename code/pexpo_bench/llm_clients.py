"""Unified LLM client wrapper.

Supports:
  • OpenAI-compatible API (GPT-4o, GPT-4o-mini, and any OpenAI-compatible endpoint
    such as DeepSeek, Together, Fireworks, or a local vLLM server).
  • Anthropic (Claude).
  • Local HuggingFace / vLLM (via OpenAI-compatible server at localhost:8000).

All backends return the same Response dataclass so the orchestrator is
model-agnostic. Token accounting is mandatory — it feeds Layer-3 cost metrics.
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

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None  # type: ignore


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
    # ---- Primary: GPT-5.4 on the NATIVE OpenAI endpoint ----
    # DeepSeek endpoints are active. gpt-5.4 now routes NATIVE (supersedes the earlier
    # do-not-failover note). This also removes the proxy as a reproducibility confound
    # for the frozen rerun; record the endpoint change in the run manifest, since the
    "gpt-5.4":       {"backend": "openai_compat", "model": "gpt-5.4",
                      "base_url": os.environ.get("OPENAI_BASE_URL_NATIVE", "https://api.openai.com/v1"),
                      "_native_openai": True,
                      "price_in": 2.5, "price_out": 15.0},
    # gpt-5.4-nano: uses NATIVE OpenAI endpoint with separate sk-proj-... key
    "gpt-5.4-nano":  {"backend": "openai_compat", "model": "gpt-5.4-nano",
                      "base_url": os.environ.get("OPENAI_BASE_URL_NATIVE", "https://api.openai.com/v1"),
                      "price_in": 0.20, "price_out": 1.25,
                      "_native_openai": True},
    # gpt-5.4-mini: NATIVE OpenAI endpoint (same key path as nano). Confirmed live —
    # model id resolves to gpt-5.4-mini-2026-03-17. price_* are estimates; recompute
    # cost from real token usage after the run.
    "gpt-5.4-mini":  {"backend": "openai_compat", "model": "gpt-5.4-mini",
                      "base_url": os.environ.get("OPENAI_BASE_URL_NATIVE", "https://api.openai.com/v1"),
                      "price_in": 0.25, "price_out": 2.00,
                      "_native_openai": True},
    "gpt-5.4-native": {"backend": "openai_compat", "model": "gpt-5.4",
                      "base_url": os.environ.get("OPENAI_BASE_URL_NATIVE", "https://api.openai.com/v1"),
                      "price_in": 2.5, "price_out": 15.0,
                      "_native_openai": True},
    "gpt-5":         {"backend": "openai_compat", "model": "gpt-5",
                      "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                      "price_in": 5.0, "price_out": 15.0},
    "gpt-4o":        {"backend": "openai_compat", "model": "gpt-4o-2024-11-20",
                      "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                      "price_in": 2.5, "price_out": 10.0},
    # gpt-4o-mini is the HR grounding judge for DeepSeek outputs — also NATIVE now
    "gpt-4o-mini":   {"backend": "openai_compat", "model": "gpt-4o-mini",
                      "base_url": os.environ.get("OPENAI_BASE_URL_NATIVE", "https://api.openai.com/v1"),
                      "_native_openai": True,
                      "price_in": 0.15, "price_out": 0.6},
    "gpt-4.1":       {"backend": "openai_compat", "model": "gpt-4.1",
                      "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                      "price_in": 2.0, "price_out": 8.0},
    # ---- Gemini via proxy ----
    "gemini-2.5-flash": {"backend": "openai_compat", "model": "gemini-2.5-flash",
                         "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                         "price_in": 0.15, "price_out": 0.6},
    "gemini-2.0-flash": {"backend": "openai_compat", "model": "gemini-2.0-flash",
                         "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                         "price_in": 0.10, "price_out": 0.4},
    # ---- DeepSeek (native API) ----
    "deepseek":      {"backend": "openai_compat", "model": "deepseek-chat",
                      "base_url": "https://api.deepseek.com",
                      "price_in": 0.27, "price_out": 1.1},
    # DeepSeek V4 — same credentials as the deepseek entry above; only the model name changes.
    # Per DeepSeek deprecation notice: legacy alias "deepseek-chat" stops 2026-07-24.
    # Flash (standard): $0.14 / $0.28 per 1M. Pro (reasoning, promo $0.435/$0.87 till 2026-05-31, regular $1.74/$3.48).
    "deepseek-v4":      {"backend": "openai_compat", "model": "deepseek-v4-flash",
                         "base_url": "https://api.deepseek.com",
                         "price_in": 0.14, "price_out": 0.28},
    "deepseek-v4-pro":  {"backend": "openai_compat", "model": "deepseek-v4-pro",
                         "base_url": "https://api.deepseek.com",
                         "price_in": 1.74, "price_out": 3.48},
    # ---- Kimi / Moonshot (native API) ----
    "kimi-k2":       {"backend": "openai_compat", "model": "kimi-k2-0711-preview",
                      "base_url": "https://api.moonshot.cn/v1",
                      "price_in": 0.5, "price_out": 2.0},
    # ---- Anthropic ----
    "claude-4-6":    {"backend": "anthropic", "model": "claude-opus-4-6",
                      "price_in": 15.0, "price_out": 75.0},
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
                    self._env_key(), "EMPTY"  # vLLM accepts anything
                )
            else:
                kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
            self._client = OpenAI(timeout=60.0, max_retries=3, **kwargs)
        elif backend == "anthropic":
            if anthropic is None:
                raise ImportError("pip install anthropic")
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _env_key(self) -> str:
        base = self.cfg.get("base_url", "")
        # Native OpenAI uses a separate project-scoped key
        if self.cfg.get("_native_openai") or base == "https://api.openai.com/v1":
            return "OPENAI_API_KEY_NATIVE"
        if "deepseek" in base:
            return "deepseek_API_KEY"
        if "moonshot" in base:
            return "KIMI_API_KEY"
        if "proxy" in base:
            return "OPENAI_API_KEY"
        if "together" in base:
            return "TOGETHER_API_KEY"
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
            # Native gpt-5.x are reasoning models: the budget is shared by hidden
            # reasoning tokens + visible answer. A 2048 cap is consumed entirely by
            # reasoning, leaving an empty answer (80%+ parse_error). Give reasoning
            # models a large floor so the JSON answer still fits.
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
                # instead of silently recording 1,104 empty answers (audit B5, fix 2026-08-12).
                name = type(e).__name__
                if "Authentication" in name or "PermissionDenied" in name or getattr(e, "status_code", None) in (401, 402, 403):
                    raise
                # A quota-exhausted 429 is a dead account, not a transient rate
                # limit — every retry returns the same error (fix 2026-08-18c:
                # OpenAI credit_balance_exhausted silently emptied a judge pass).
                if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                    raise
                # content-filter / invalid_prompt / transient API error: return an
                # empty response so the item is recorded as a failure (parse_error)
                # rather than crashing the whole run.
                content, in_tok, out_tok, finish = "", 0, 0, f"error:{name}"
        elif backend == "anthropic":
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            other = [m for m in messages if m["role"] != "system"]
            resp = self._client.messages.create(
                model=self.cfg["model"],
                system=system_msg,
                messages=other,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            content = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            )
            in_tok = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
            finish = resp.stop_reason
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


def get_client(model_key: str = "gpt-4o") -> LLMClient:
    global _default_client
    if _default_client is None or _default_client.model_key != model_key:
        _default_client = LLMClient(model_key)
    return _default_client
