"""Unified orchestrator for the 5 architectures (A0–A4).

Design principles:
  • Same interface: each architecture exposes `run(question: dict) -> Result`.
  • Same LLM call wrapper so token / latency accounting is comparable.
  • Differences are isolated in build_messages() and the tool/retrieval loop.

This is a reference implementation skeleton — concrete LLM client and
vector store calls are indicated by `# TODO` markers.
"""
from __future__ import annotations

import json
import pathlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pexpo_bench.architectures.prompts import (
    A0_SYSTEM, A1_SYSTEM, A2_SYSTEM, A2P_SYSTEM, A2_USER_TEMPLATE,
    A3_SYSTEM, A4_SYSTEM, A4P_SYSTEM,
)
from pexpo_bench.tools import TOOL_REGISTRY  # unified: base + health + meta


# ==========================================================================
# Data classes
# ==========================================================================
@dataclass
class ToolCall:
    tool: str
    args: dict
    output: Any = None
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class Result:
    qid: str
    architecture: str
    answer: Any
    unit: str | None
    reasoning: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    retrieved_docs: list[dict] = field(default_factory=list)
    raw_output: str = ""
    # Cost & latency
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_s: float = 0.0
    # Error
    parse_error: bool = False
    error_msg: str | None = None


# ==========================================================================
# Rate-limit aware retry ()
# ==========================================================================
def _call_with_rate_limit_retry(client, kwargs, max_retries=6,
                                base_delay=8.0, max_delay=120.0):
    """Wrap OpenAI chat.completions.create with explicit 429 backoff.

    OpenAI SDK's built-in retry is not enough for TPM caps (e.g. nano 200k TPM).
    Exponential backoff: 8s, 16s, 32s, 64s, 120s, 120s. Total wait up to ~6 min.
    """
    import openai as _openai_mod
    last_exc = None
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except _openai_mod.RateLimitError as e:
            last_exc = e
            # Parse Retry-After if available
            wait = delay
            ra = getattr(e, "response", None)
            if ra is not None:
                ra_h = getattr(ra, "headers", {}) or {}
                rh = ra_h.get("retry-after") or ra_h.get("Retry-After")
                try:
                    if rh: wait = max(wait, float(rh))
                except Exception:
                    pass
            wait = min(wait, max_delay)
            time.sleep(wait)
            delay = min(delay * 2, max_delay)
        except (_openai_mod.APIConnectionError,
                _openai_mod.APITimeoutError) as e:
            last_exc = e
            time.sleep(min(delay, 30.0))
            delay = min(delay * 2, max_delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry exhausted with no captured exception")


# ==========================================================================
# LLM wrapper (: model_key-dispatched via LLMClient)
# ==========================================================================
def llm_call(messages: list[dict], model_key: str = "gpt-5.4",
             temperature: float = 0.3, max_tokens: int = 2048,
             seed: int | None = 42) -> dict:
    """ LLM call via the unified LLMClient.

    Returns {"content": str, "input_tokens": int, "output_tokens": int,
             "latency_s": float, "finish_reason": str}.
    """
    from pexpo_bench.llm_clients import LLMClient
    client = LLMClient(model_key, temperature=temperature,
                       max_tokens=max_tokens, seed=seed)
    r = client.chat(messages)
    return {
        "content": r.content,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "latency_s": r.latency_s,
        "finish_reason": r.finish_reason,
    }


# ==========================================================================
# Retrieval wrapper (: dedup + low-info filter via Retriever)
# ==========================================================================
_RETRIEVER_CACHE = None
_RETRIEVER_LOCK = threading.Lock()
def retrieve(query: str, k: int = 5,
             mode: str = "rerank") -> list[dict]:
    """ hybrid retriever: BM25 (keyword) + dense (semantic) + cross-encoder rerank
    + content dedup + low-info filter.

    Per RAG optimization: environmental chemical names (arsenic, RfD, MCL,
    NO2, PM2.5, formaldehyde, IUR) need keyword matching which pure dense
    vector search may miss. Hybrid BM25+dense + cross-encoder rerank is the
    standard solution. Mode 'rerank' = full pipeline.

    Returns list of {doc_id, section, text, score, chunk_id}.
    """
    global _RETRIEVER_CACHE
    with _RETRIEVER_LOCK:  # check-then-set was racy under the threaded runner (fix 2026-08-12)
        if _RETRIEVER_CACHE is None:
            from pexpo_bench.retrieval import Retriever
            _index_dir = pathlib.Path(__file__).resolve().parents[1] / "knowledge_base" / "index"
            _RETRIEVER_CACHE = Retriever.load(
                str(_index_dir),  # was a stale absolute path to ~/Desktop/lzl (fix 2026-08-12)
                embed_model_name="all-MiniLM-L6-v2",
                reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # small, fast cross-encoder
                use_bm25=True,
            )
    # 2026-08-17: retrieve() must be serialized — SentenceTransformer/CrossEncoder/faiss
    # are not thread-safe under the concurrent runner (native crash, no traceback, when
    # 10 worker threads query simultaneously). LLM calls remain parallel; only the
    # retrieval step queues here (~0.5-1 s per query).
    with _RETRIEVER_LOCK:
        passages = _RETRIEVER_CACHE.retrieve(query, k=k, mode=mode)
    return [
        {"doc_id": p.doc_id, "section": p.section, "text": p.text,
         "score": p.score, "chunk_id": p.chunk_id}
        for p in passages
    ]


# ==========================================================================
# Output parser
# ==========================================================================
def parse_json_output(raw: str) -> dict:
    """Robust JSON parse — strip code fences if any."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


_NUM_UNIT_RE = re.compile(
    r"""
    [-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d+)?(?:[eE][-+]?\d+)?
    \s*
    (?P<unit>[A-Za-zµμ][\w\^/³²·\-]*)?
    """,
    re.VERBOSE,
)


def _extract_answer_unit(text: str, qtype: str | None):
    """Promote `FINAL: ...` text into typed (answer, unit).

    T/F → bool; calc → (number, unit_str); open-ended → trimmed text."""
    # Strip common trailing markdown/punctuation (e.g. "True**", "False`", "True.")
    s = text.strip().rstrip("*`'\".,;:!?#-")
    low = s.lower()
    if qtype == "true_false":
        first = low.split()[0] if low else ""
        first = first.rstrip("*`'\".,;:!?#-")
        if first in ("true", "yes", "correct"):
            return True, None
        if first in ("false", "no", "incorrect"):
            return False, None
        return s[:200], None
    if qtype == "calculation":
        m = _NUM_UNIT_RE.search(s)
        if m:
            num_raw = m.group(0).split()[0].replace(",", "")
            try:
                value = float(num_raw)
                if value.is_integer() and "." not in num_raw and "e" not in num_raw.lower():
                    value = int(value)
                unit = m.group("unit")
                return value, unit
            except ValueError:
                pass
        return s[:200], None
    # open-ended: keep full text (cap at 4000 to bound storage)
    return s[:4000], None


# ==========================================================================
# Base runner
# ==========================================================================
class BaseArch:
    name: str = "base"
    system_prompt: str = ""

    def __init__(self, model_key: str = "gpt-5.4",
                 temperature: float = 0.3, max_tokens: int = 2048,
                 seed: int | None = 42):
        self.model_key = model_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def build_messages(self, question: dict) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question["question"]},
        ]

    def run(self, question: dict) -> Result:
        t0 = time.time()
        messages = self.build_messages(question)
        resp = llm_call(messages, model_key=self.model_key,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens, seed=self.seed)
        parsed, parse_err = self._safe_parse(resp["content"])
        return Result(
            qid=question["qid"],
            architecture=self.name,
            answer=parsed.get("answer"),
            unit=parsed.get("unit"),
            reasoning=parsed.get("reasoning", ""),
            citations=parsed.get("citations", []),
            raw_output=resp["content"],
            input_tokens=resp["input_tokens"],
            output_tokens=resp["output_tokens"],
            total_latency_s=time.time() - t0,
            parse_error=parse_err,
        )

    @staticmethod
    def _safe_parse(raw: str) -> tuple[dict, bool]:
        try:
            return parse_json_output(raw), False
        except Exception:
            return {}, True


# ==========================================================================
# A0 — Naive
# ==========================================================================
class A0_Naive(BaseArch):
    name = "A0_naive"
    system_prompt = A0_SYSTEM


# ==========================================================================
# A1 — Context Engineering
# ==========================================================================
class A1_ContextEng(BaseArch):
    name = "A1_context_eng"
    system_prompt = A1_SYSTEM


# ==========================================================================
# A2 — RAG
# ==========================================================================
class A2_RAG(BaseArch):
    name = "A2_rag"
    system_prompt = A2_SYSTEM
    top_k: int = 5

    def _get_passages(self, question: dict) -> list[dict]:
        """Passage source for the RAG arm. Overridden by the oracle arm."""
        return retrieve(question["question"], k=self.top_k)

    def run(self, question: dict) -> Result:
        t0 = time.time()
        passages = self._get_passages(question)
        passages_str = "\n\n".join(
            f"[{p['doc_id']} | {p['section']}] {p['text']}" for p in passages
        )
        user_msg = A2_USER_TEMPLATE.format(
            passages=passages_str, question=question["question"]
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        resp = llm_call(messages, model_key=self.model_key,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens, seed=self.seed)
        parsed, parse_err = self._safe_parse(resp["content"])
        return Result(
            qid=question["qid"],
            architecture=self.name,
            answer=parsed.get("answer"),
            unit=parsed.get("unit"),
            reasoning=parsed.get("reasoning", ""),
            citations=parsed.get("citations", []),
            retrieved_docs=passages,
            raw_output=resp["content"],
            input_tokens=resp["input_tokens"],
            output_tokens=resp["output_tokens"],
            total_latency_s=time.time() - t0,
            parse_error=parse_err,
        )


# A2+ — RAG with tightened evidence-use constraints.
# Same pipeline as A2_RAG but with a prompt that forbids retrieved
# authoritative values from displacing question-embedded scenario parameters.
class A2P_RAG_Constrained(A2_RAG):
    name = "A2p_rag_constrained"
    system_prompt = A2P_SYSTEM


# ==========================================================================
# A3 — Agentic Harness (ReAct loop)
# ==========================================================================
class A3_Agent(BaseArch):
    """Agentic Harness using OpenAI-native function calling (not text-based ReAct)."""
    name = "A3_agent"
    system_prompt = A3_SYSTEM
    max_steps: int = 8

    # ========================================================================
    # TOOL_DEFS — 16 tools for A3 (A4 adds `retrieve` = 17 total)
    # Submit_plan must be called FIRST (enforced via prompt + first-tool check).
    # ========================================================================
    TOOL_DEFS = [
        # ---- META: planning ----
        {"type": "function", "function": {
            "name": "submit_plan",
            "description": "REQUIRED first call. Submit a ≥2-step plan describing how you intend to solve this question (which lookups, which calculations, which checks).",
            "parameters": {"type": "object", "properties": {
                "steps": {"type": "array", "items": {"type": "string"},
                          "description": "Ordered plan steps in natural language"},
                "expected_tools": {"type": "array", "items": {"type": "string"},
                                   "description": "Tool names you anticipate calling (optional)"}
            }, "required": ["steps"]}}},
        {"type": "function", "function": {
            "name": "revise_plan",
            "description": "Call when a previous tool result invalidates your plan. Provide a reason and new steps.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "description": "Why the plan needs revision (≥10 chars)"},
                "new_steps": {"type": "array", "items": {"type": "string"}}
            }, "required": ["reason", "new_steps"]}}},
        # ---- COMPUTATION / GENERAL ----
        {"type": "function", "function": {
            "name": "python_sandbox",
            "description": "Execute restricted Python (math/numpy/statistics) for any numerical computation. 5-second timeout.",
            "parameters": {"type": "object", "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            }, "required": ["code"]}}},
        {"type": "function", "function": {
            "name": "unit_converter",
            "description": "Convert between units (min↔h, L↔mL, µg/m³↔mg/m³, ng↔µg↔mg, etc.)",
            "parameters": {"type": "object", "properties": {
                "value": {"type": "number"},
                "from_": {"type": "string", "description": "Source unit"},
                "to": {"type": "string", "description": "Target unit"}
            }, "required": ["value", "from_", "to"]}}},
        # ---- EXPOSURE FACTORS / DOSIMETRY ----
        {"type": "function", "function": {
            "name": "exposure_factor_lookup",
            "description": "EPA Exposure Factors Handbook (EFH 2011) default values: inhalation_rate, body_weight, skin_area, drinking_water, soil_ingestion, time_indoors/outdoors by age and sex.",
            "parameters": {"type": "object", "properties": {
                "factor": {"type": "string", "description": "Factor name"},
                "age": {"description": "Age in years (number) or group ('adult','child','infant', '<1','1-2','3-6','6-11','11-16','16-21')"},
                "sex": {"type": "string", "enum": ["both", "M", "F"], "default": "both"},
                "duration": {"type": "string", "default": "long_term"}
            }, "required": ["factor"]}}},
        {"type": "function", "function": {
            "name": "dose_calculator",
            "description": "Dose = C × IR × ET (÷ BW if given). Use for inhalation/ingestion intake calc.",
            "parameters": {"type": "object", "properties": {
                "C": {"type": "number", "description": "Concentration (µg/m³ or mg/L)"},
                "IR": {"type": "number", "description": "Intake rate (m³/h or L/day)"},
                "ET": {"type": "number", "description": "Exposure time (h or day)"},
                "BW": {"type": "number", "description": "Body weight kg (optional)"}
            }, "required": ["C", "IR", "ET"]}}},
        {"type": "function", "function": {
            "name": "mppd_deposition",
            "description": "ICRP-66 / MPPD respiratory deposition fraction for a particle size in head/TB/alveolar/total.",
            "parameters": {"type": "object", "properties": {
                "particle_size_um": {"type": "number"},
                "region": {"type": "string", "enum": ["head", "TB", "alveolar", "total"]},
                "breathing": {"type": "string", "enum": ["resting", "light", "heavy"], "default": "resting"}
            }, "required": ["particle_size_um", "region"]}}},
        # ---- MICROENVIRONMENT ----
        {"type": "function", "function": {
            "name": "indoor_air_mass_balance",
            "description": "Steady-state indoor concentration: C_in = (P·AER·C_out + S/V) / (AER + k_dep). When k_dep=0 and S=0 the AER cancels and C_in = P·C_out.",
            "parameters": {"type": "object", "properties": {
                "C_out": {"type": "number", "description": "Outdoor concentration"},
                "P": {"type": "number", "default": 1.0, "description": "Penetration factor (0-1)"},
                "AER": {"type": "number", "default": 0.5, "description": "Air exchange rate h⁻¹"},
                "k_dep": {"type": "number", "default": 0.0, "description": "Indoor deposition rate h⁻¹"},
                "S": {"type": "number", "default": 0.0, "description": "Indoor source emission rate (mass/h)"},
                "V": {"type": "number", "default": 100.0, "description": "Room volume m³"}
            }, "required": ["C_out"]}}},
        {"type": "function", "function": {
            "name": "airquality_lookup",
            "description": "Cached AirNow/PurpleAir lookup (stub; returns default values).",
            "parameters": {"type": "object", "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "time": {"type": "string", "description": "ISO-8601 timestamp"},
                "pollutant": {"type": "string", "enum": ["PM2.5","PM10","NO2","O3","CO","SO2"]}
            }, "required": ["lat", "lon", "time", "pollutant"]}}},
        {"type": "function", "function": {
            "name": "trajectory_match",
            "description": "Match GPS trajectory CSV → microenvironment segments with duration.",
            "parameters": {"type": "object", "properties": {
                "gps_csv": {"type": "string", "description": "Path or identifier of GPS CSV"},
                "microenv_db": {"type": "string", "default": "default"}
            }, "required": ["gps_csv"]}}},
        # ---- TOXICOLOGY / GUIDELINES ----
        {"type": "function", "function": {
            "name": "iris_lookup",
            "description": "EPA IRIS toxicity value lookup. 10 chemicals supported: benzene, formaldehyde, arsenic, lead, cadmium, chromium_vi, benzo_a_pyrene, toluene, manganese, cotinine.",
            "parameters": {"type": "object", "properties": {
                "chemical": {"type": "string", "description": "chemical name or alias"},
                "value_type": {"type": "string", "enum": ["RfD","RfC","IUR","oral_CSF"], "default": "RfD"}
            }, "required": ["chemical"]}}},
        {"type": "function", "function": {
            "name": "who_aqg_lookup",
            "description": "WHO Air Quality Guideline values (2021 default; pass version='2005' for previous). Returns the guideline µg/m³ for the pollutant + averaging window.",
            "parameters": {"type": "object", "properties": {
                "pollutant": {"type": "string", "enum": ["PM2.5","PM10","NO2","O3","SO2","CO"]},
                "window": {"type": "string", "description": "annual | 24h | 8h | peak-season", "default": "annual"},
                "version": {"type": "string", "enum": ["2021","2005"], "default": "2021"}
            }, "required": ["pollutant"]}}},
        # ---- HEALTH RISK ----
        {"type": "function", "function": {
            "name": "af_calc",
            "description": "Attributable fraction. AF = (RR-1)/RR; optionally PAF = p(RR-1)/(1+p(RR-1)) if prevalence_exposed given.",
            "parameters": {"type": "object", "properties": {
                "RR": {"type": "number", "description": "Relative risk"},
                "prevalence_exposed": {"type": "number", "description": "Fraction exposed in population [0,1] (optional, for PAF)"}
            }, "required": ["RR"]}}},
        {"type": "function", "function": {
            "name": "gbd_mortality",
            "description": "Attributable cases/deaths per year. cases = population × (baseline_per_100k/100k) × AF.",
            "parameters": {"type": "object", "properties": {
                "population": {"type": "number"},
                "baseline_rate_per_100k": {"type": "number", "description": "Cases per 100,000 per year"},
                "AF": {"type": "number", "description": "Attributable fraction (preferred)"},
                "RR": {"type": "number", "description": "If AF not given, AF computed from (RR-1)/RR"}
            }, "required": ["population", "baseline_rate_per_100k"]}}},
        {"type": "function", "function": {
            "name": "ier_pm25_rr",
            "description": "GBD 2019 IER-based RR for PM2.5. Endpoints: IHD, stroke, COPD, LC, LRI. TMREL default 5 µg/m³ (range 2.4-5.9).",
            "parameters": {"type": "object", "properties": {
                "C": {"type": "number", "description": "PM2.5 concentration µg/m³"},
                "endpoint": {"type": "string", "enum": ["IHD","stroke","COPD","LC","LRI"]},
                "tmrel": {"type": "number", "default": 5.0}
            }, "required": ["C", "endpoint"]}}},
        {"type": "function", "function": {
            "name": "noncancer_hq_calc",
            "description": "Hazard Quotient = exposure / reference. Use RfC for inhalation, RfD for oral.",
            "parameters": {"type": "object", "properties": {
                "exposure": {"type": "number"},
                "reference": {"type": "number"},
                "route": {"type": "string", "enum": ["inhalation","oral"], "default": "inhalation"}
            }, "required": ["exposure", "reference"]}}},
        {"type": "function", "function": {
            "name": "cotinine_pk_calc",
            "description": "Convert blood/plasma cotinine concentration (ng/mL) + blood volume (L) to total body amount.",
            "parameters": {"type": "object", "properties": {
                "concentration_ng_per_mL": {"type": "number"},
                "blood_volume_L": {"type": "number", "default": 5.0},
                "output_unit": {"type": "string", "enum": ["ng","µg","mg"], "default": "µg"}
            }, "required": ["concentration_ng_per_mL"]}}},
    ]

    def __init__(self, model_key: str = "gpt-5.4",
                 temperature: float = 0.3, max_tokens: int = 2048,
                 seed: int | None = 42):
        """: accepts model_key dispatched via MODEL_REGISTRY."""
        super().__init__() if hasattr(super(), "__init__") else None
        self.model_key = model_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        from pexpo_bench.llm_clients import MODEL_REGISTRY
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model_key: {model_key}")
        self.cfg = MODEL_REGISTRY[model_key]

    def run(self, question: dict) -> Result:
        from openai import OpenAI
        import os
        t0 = time.time()

        # dispatch: pick correct env var based on model config
        base = self.cfg["base_url"]
        if self.cfg.get("_native_openai") or base == "https://api.openai.com/v1":
            api_key = os.environ.get("OPENAI_API_KEY_NATIVE")
        elif "deepseek" in base:
            api_key = os.environ.get("deepseek_API_KEY")
        elif "moonshot" in base:
            api_key = os.environ.get("KIMI_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

        oai = OpenAI(api_key=api_key, base_url=base, timeout=120.0, max_retries=2)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question["question"]},
        ]
        tool_calls_result: list[ToolCall] = []
        in_tok = out_tok = 0

        # Token kwarg name varies: native OpenAI 5.x requires max_completion_tokens.
        # For native reasoning models the cap must include hidden reasoning tokens; a bare
        # 2048 is consumed entirely by reasoning and yields empty answers (same floor
        # LLMClient.chat applies for A0-A2; agent-path parity fix 2026-08-12).
        if self.cfg.get("_native_openai"):
            tokens_kwarg = 'max_completion_tokens'
            tokens_value = max(self.max_tokens, 16384)
        else:
            tokens_kwarg = 'max_tokens'
            tokens_value = self.max_tokens

        for step in range(self.max_steps):
            call_kwargs = {
                'model': self.cfg["model"],
                'messages': messages,
                'tools': self.TOOL_DEFS,
                'tool_choice': "auto",
                'temperature': self.temperature,
                tokens_kwarg: tokens_value,
            }
            if self.seed is not None and not self.cfg.get("_native_openai"):
                call_kwargs['seed'] = self.seed
            resp = _call_with_rate_limit_retry(oai, call_kwargs)
            msg = resp.choices[0].message
            in_tok += resp.usage.prompt_tokens
            out_tok += resp.usage.completion_tokens

            # If model made tool calls
            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    executed = self._execute_tool({"tool": fn_name, "args": fn_args})
                    tool_calls_result.append(executed)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(executed.output, default=str),
                    })
                continue  # next iteration to get final answer

            # No tool calls — model is done
            content = msg.content or ""
            parsed, parse_err = self._safe_parse(content)
            if parse_err:
                # Agent prompt asks for `FINAL: ...` plain text, not JSON.
                # Extract everything after the last "FINAL:" marker, then
                # promote T/F bools and pull out leading numeric+unit for calc.
                final_text = content
                if "FINAL:" in content:
                    final_text = content.rsplit("FINAL:", 1)[1].strip()
                ans, unit = _extract_answer_unit(final_text,
                                                 question.get("question_type"))
                parsed = {"answer": ans, "unit": unit, "reasoning": content}
                parse_err = False
            return Result(
                qid=question["qid"], architecture=self.name,
                answer=parsed.get("answer"), unit=parsed.get("unit"),
                reasoning=parsed.get("reasoning", content),
                citations=parsed.get("citations", []),
                tool_calls=tool_calls_result,
                raw_output=content,
                input_tokens=in_tok, output_tokens=out_tok,
                total_latency_s=time.time() - t0,
                parse_error=parse_err,
            )

        return Result(
            qid=question["qid"], architecture=self.name,
            answer=None, unit=None, reasoning="max_steps_exceeded",
            tool_calls=tool_calls_result,
            input_tokens=in_tok, output_tokens=out_tok,
            total_latency_s=time.time() - t0,
            parse_error=True, error_msg="max_steps_exceeded",
        )

    @staticmethod
    def _execute_tool(action: dict) -> ToolCall:
        name, args = action.get("tool"), action.get("args", {})
        t0 = time.time()
        # Normalize kwarg name for unit_converter
        if name == "unit_converter" and "from" in args:
            args["from_"] = args.pop("from")
        fn: Callable | None = TOOL_REGISTRY.get(name)
        if fn is None:
            return ToolCall(tool=name, args=args, error=f"unknown tool: {name}",
                            latency_s=time.time() - t0)
        try:
            out = fn(**args)
            return ToolCall(tool=name, args=args, output=out,
                            latency_s=time.time() - t0)
        except Exception as e:
            return ToolCall(tool=name, args=args, error=str(e),
                            latency_s=time.time() - t0)


# ==========================================================================
# A4 — Hybrid (Agent + Retrieval as a tool)
# ==========================================================================
class A4_Hybrid(A3_Agent):
    name = "A4_hybrid"
    system_prompt = A4_SYSTEM
    max_steps: int = 10

    # Inherits __init__ from A3_Agent (model_key dispatch).

    # : A4 = A3's 16 tools + retrieve (17 total)
    TOOL_DEFS = A3_Agent.TOOL_DEFS + [
        {"type": "function", "function": {
            "name": "retrieve",
            "description": (
                "Search the PEA knowledge base (1103 documents incl. EPA EFH, "
                "WHO AQG, IRIS, ATSDR profiles, ICRP, and peer-reviewed "
                "methodology papers). Returns top-k content-deduped passages. "
                "Use when the question requires an authoritative reference "
                "value or a methodology specification."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "k": {"type": "integer", "default": 5, "description": "Number of passages (1-10)"}
            }, "required": ["query"]}}},
    ]

    @staticmethod
    def _execute_tool(action: dict) -> ToolCall:
        if action.get("tool") == "retrieve":
            t0 = time.time()
            try:
                passages = retrieve(**action.get("args", {}))
                return ToolCall(tool="retrieve", args=action["args"],
                                output=passages, latency_s=time.time() - t0)
            except Exception as e:
                return ToolCall(tool="retrieve", args=action.get("args", {}),
                                error=str(e), latency_s=time.time() - t0)
        return A3_Agent._execute_tool(action)


# A4+ — Hybrid with the same tightened evidence-use rules applied to the
# retrieve() tool (parallel to A2+).
class A4P_Hybrid_Constrained(A4_Hybrid):
    name = "A4p_hybrid_constrained"
    system_prompt = A4P_SYSTEM


# ==========================================================================
# A3_AEGEA — Adaptive Evidence-Gated Expert Agent ( main A3)
# Replaces basic A3_Agent (which is kept as A3_BasicToolUse for SI comparison).
# ==========================================================================
class A3_AEGEA(BaseArch):
    """4-module routing pipeline:
       Scenario Parser → Task Router → (Hybrid Retrieve → Evidence Gate)
       → per-route LLM call (DIRECT / RAG / CALCULATOR / RAG_CALCULATOR / SAFETY_LIMITED)"""
    name = "A3_agent"  # registry key stays A3_agent so existing analysis works

    def __init__(self, model_key: str = "gpt-5.4", temperature: float = 0.3,
                 max_tokens: int = 2048, seed: int | None = 42):
        super().__init__(model_key=model_key, temperature=temperature,
                         max_tokens=max_tokens, seed=seed)

    def run(self, question: dict) -> Result:
        from pexpo_bench.architectures.aegea import run_aegea
        t0 = time.time()
        try:
            r = run_aegea(question["question"],
                          main_model_key=self.model_key,
                          temperature=self.temperature,
                          max_tokens=self.max_tokens)
            return Result(
                qid=question["qid"], architecture=self.name,
                answer=r.answer, unit=r.unit,
                reasoning=r.reasoning, citations=r.citations,
                retrieved_docs=r.retrieved,
                tool_calls=[],  # AEGEA's tool_use is encoded in 'route' field
                raw_output=r.raw_output,
                input_tokens=r.in_tokens, output_tokens=r.out_tokens,
                total_latency_s=time.time() - t0,
                parse_error=r.parse_error,
                error_msg=f"route={r.route}",  # encode route in error_msg field
            )
        except Exception as e:
            import traceback
            return Result(
                qid=question["qid"], architecture=self.name,
                answer=None, unit=None,
                reasoning="", citations=[], retrieved_docs=[], tool_calls=[],
                total_latency_s=time.time() - t0,
                parse_error=True,
                error_msg=f"AEGEA exception: {type(e).__name__}: {str(e)[:200]}",
            )


# A3_Agent (basic tool-use) is canonical A3; A3_AEGEA kept only as SI exploration.
A3_Agent.name = "A3_agent"
A3_AEGEA.name = "A3_aegea"


# ==========================================================================
# Oracle-retrieval arms (perfect-recall upper bound)
# ==========================================================================
# The deployed retriever surfaces the gold passage in the top-5 for only ~41%
# of items, so "retrieval adds nothing" is confounded with retriever quality.
# These arms replace retrieval with the item's own gold_references chunk, i.e.
# recall is 100% by construction. They bound what retrieval could contribute.
def _gold_passages(question: dict, k: int = 5) -> list[dict]:
    """Gold passages for the oracle arm.

    The programmatic-template stream (n = 100) stores gold_references as bare
    citation strings with no quoted text, so the oracle is undefined there and
    those items fall back to the real retriever.
    """
    out = []
    for g in (question.get("gold_references") or [])[:k]:
        if isinstance(g, dict) and g.get("quote"):
            out.append({"doc_id": g.get("doc_id", ""),
                        "section": f"p.{g.get('page', '')}",
                        "text": g.get("quote", ""),
                        "score": 1.0,
                        "chunk_id": g.get("chunk_id", "")})
    return out


class A2_Oracle(A2P_RAG_Constrained):
    """A2+ with the item's own gold passage injected as RETRIEVED_CONTEXT."""
    name = "A2_oracle"

    def _get_passages(self, question: dict) -> list[dict]:
        return _gold_passages(question, k=self.top_k)


class A4_Oracle(A4P_Hybrid_Constrained):
    """A4+ whose retrieve() tool always returns the item's own gold passage.

    2026-08-12 fix: the current question is kept in a threading.local. The previous
    `self._gold_q` attribute was shared across the runner's worker threads, so with
    concurrency >= 2 a question could receive ANOTHER question's gold passages (audit B8).
    """
    name = "A4_oracle"
    _tls = threading.local()

    def run(self, question: dict) -> Result:
        self._tls.gold_q = question      # per-thread: A3 loop calls self._execute_tool
        try:
            return super().run(question)
        finally:
            self._tls.gold_q = None

    def _execute_tool(self, action: dict) -> ToolCall:
        if action.get("tool") == "retrieve":
            t0 = time.time()
            return ToolCall(tool="retrieve", args=action.get("args", {}),
                            output=_gold_passages(getattr(self._tls, "gold_q", None) or {}),
                            latency_s=time.time() - t0)
        return A3_Agent._execute_tool(action)


# ==========================================================================
# Factorial arms (Phase B rerun, 2026-08-13): decompose the A4p−A3 contrast into
# R = retrieval availability, P = evidence-use rules, B = step budget (10 vs 8).
# Cube corners already present: A3_Agent = (R0,P0,B0); A4_Hybrid = (R1,P0,B1);
# A4P_Hybrid_Constrained = (R1,P1,B1). The five classes below complete the 2x2x2.
# Known minor prompt confound, documented in RERUN_PROTOCOL: R-arms inherit the
# A4/A4P system text whose step-count sentence says 10 even when max_steps is 8.
# ==========================================================================
from pexpo_bench.architectures.prompts import _EVIDENCE_RULES  # noqa: E402


class F_A3_R(A4_Hybrid):                    # (R1,P0,B0)
    name = "fA3_R"
    max_steps = 8


class F_A3_P(A3_Agent):                     # (R0,P1,B0)
    name = "fA3_P"
    system_prompt = A3_SYSTEM + "\n\n" + _EVIDENCE_RULES


class F_A3_B(A3_Agent):                     # (R0,P0,B1)
    name = "fA3_B"
    max_steps = 10


class F_A3_RP(A4P_Hybrid_Constrained):      # (R1,P1,B0)
    name = "fA3_RP"
    max_steps = 8


class F_A3_PB(A3_Agent):                    # (R0,P1,B1)
    name = "fA3_PB"
    system_prompt = A3_SYSTEM + "\n\n" + _EVIDENCE_RULES
    max_steps = 10


# ==========================================================================
# Registry
# ==========================================================================
ARCHITECTURES = {
    "fA3_R":                     F_A3_R,             # factorial: retrieval only
    "fA3_P":                     F_A3_P,             # factorial: evidence rules only
    "fA3_B":                     F_A3_B,             # factorial: budget only
    "fA3_RP":                    F_A3_RP,            # factorial: retrieval + rules
    "fA3_PB":                    F_A3_PB,            # factorial: rules + budget
    "A0_naive":                  A0_Naive,
    "A1_context_eng":            A1_ContextEng,
    "A2_rag":                    A2_RAG,
    "A2p_rag_constrained":       A2P_RAG_Constrained,
    "A2_oracle":                 A2_Oracle,          # perfect-recall upper bound
    "A3_agent":                  A3_Agent,           # basic tool-use (canonical A3)
    "A3_aegea":                  A3_AEGEA,           # SI: negative-result variant
    "A4_hybrid":                 A4_Hybrid,
    "A4p_hybrid_constrained":    A4P_Hybrid_Constrained,
    "A4_oracle":                 A4_Oracle,          # perfect-recall upper bound
}


# ==========================================================================
# CLI entry (pilot)
# ==========================================================================
if __name__ == "__main__":
    import argparse, yaml, pathlib, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=list(ARCHITECTURES))
    parser.add_argument("--questions", default="pexpo_bench/samples/sample_questions.yaml")
    parser.add_argument("--out", default="pilot_results.jsonl")
    args = parser.parse_args()

    questions = yaml.safe_load(pathlib.Path(args.questions).read_text())
    runner = ARCHITECTURES[args.arch]()
    with open(args.out, "a") as f:
        for q in questions:
            try:
                r = runner.run(q)
            except NotImplementedError as e:
                print(f"[skip] {q['qid']}: {e}", file=sys.stderr)
                continue
            f.write(json.dumps(r.__dict__, default=lambda o: o.__dict__) + "\n")
