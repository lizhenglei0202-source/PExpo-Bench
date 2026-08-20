"""AEGEA — Adaptive Evidence-Gated Expert Agent.

 architecture for PEA. 4-module pipeline:
  1. Scenario Parser (small LLM, JSON output)
  2. Task Router (deterministic rule + small LLM fallback)
  3. Hybrid Retrieve (BM25 + dense + cross-encoder rerank)
  4. Evidence Gate (small LLM, JSON output)

Then routes to one of:
  - DIRECT : answer from knowledge, no RAG / no tools
  - RAG : retrieve → gate → grounded answer (or fallback)
  - CALCULATOR : use python_sandbox or domain tools
  - RAG_CALCULATOR : retrieve → compute with tools + grounded
  - SAFETY_LIMITED : pediatric/acute/pregnant — flag uncertainty

Key design: small LLM (DeepSeek V4 Flash) for parser/router/gate (~$0.001/q),
base model only for final reasoning. Total ~5 LLM calls per question.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

# Small LLM used for parsing / routing / gating (cheap, deterministic temp=0)
_SMALL_LLM_KEY = "deepseek-v4"


# ==========================================================================
# Module 1: Scenario Parser
# ==========================================================================
_PARSER_SYSTEM = """You are a scenario-extraction module for an environmental health agent.

Given a question, extract a structured JSON exposure scenario. Output ONLY the JSON object, no other text. Use null when info is not present.

Schema:
{
  "pollutants": ["chemical names mentioned, e.g. PM2.5, arsenic, formaldehyde"],
  "medium": ["air", "water", "soil", "food", "biomarker"],
  "exposure_routes": ["inhalation", "ingestion", "dermal"],
  "population": ["adult", "child", "elderly", "pregnant", "general"],
  "exposure_duration": "acute | subchronic | chronic | lifetime | null",
  "concentration_or_measurement": "the specific value(s) provided, or null",
  "question_type": "definition | general_risk_explanation | individual_exposure_assessment | quantitative_dose_estimation | guideline_or_threshold_question | mitigation_advice | evidence_comparison | multi_pollutant_exposure | sensitive_population | medical_or_urgent",
  "requires_quantitative_estimation": true,
  "requires_current_guideline": false,
  "requires_local_regulation": false,
  "requires_medical_advice": false
}"""


def parse_scenario(question: str, small_client) -> dict:
    """Module 1: return structured scenario JSON. Falls back to minimal dict on parse fail."""
    try:
        r = small_client.chat([
            {"role": "system", "content": _PARSER_SYSTEM},
            {"role": "user", "content": f"Question:\n{question}\n\nExtract JSON."},
        ], temperature=0.0, max_tokens=512)
        text = r.content.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"): text = text[4:]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return {"question_type": "unknown", "requires_quantitative_estimation": False}


# ==========================================================================
# Module 2: Task Router (deterministic rule, with safety fallback)
# ==========================================================================
def route_task(scenario: dict, question: str) -> dict:
    """Module 2: decide route from {DIRECT, RAG, CALCULATOR, RAG_CALCULATOR, SAFETY_LIMITED}."""
    qtype = (scenario.get("question_type") or "").lower()

    # SAFETY: pediatric / pregnant / acute / urgent
    if (scenario.get("requires_medical_advice")
        or qtype == "medical_or_urgent"
        or "pregnant" in (scenario.get("population") or [])
        or re.search(r"\b(acute|poisoning|emergency|urgent|symptom)\b", question.lower())):
        return {"route": "SAFETY_LIMITED",
                "reason": "Medical/acute/pediatric — limit to conservative info"}

    # CALCULATION needed
    needs_calc = (scenario.get("requires_quantitative_estimation")
                  or qtype in ("quantitative_dose_estimation", "individual_exposure_assessment")
                  or bool(re.search(r"\bcalculate\b|\bcompute\b|\bestimate\b", question.lower())))

    # Has concentration / parameters → calc with values
    has_params = bool(scenario.get("concentration_or_measurement")) or bool(
        re.search(r"\d+(?:\.\d+)?\s*(?:µg|μg|ug|mg|kg|m³|m3|L|h/day|hr|day|year|ppm|ppb)",
                  question))

    # Guideline / threshold → RAG
    needs_guideline = (qtype == "guideline_or_threshold_question"
                       or scenario.get("requires_current_guideline")
                       or bool(re.search(r"\bguideline\b|\bAQG\b|\bRfD\b|\bRfC\b|\bMCL\b|\bIUR\b|\bIRIS\b|\bWHO\b",
                                         question)))

    if needs_calc and (needs_guideline or has_params):
        return {"route": "RAG_CALCULATOR",
                "reason": "Quantitative + reference value lookup"}
    if needs_calc:
        return {"route": "CALCULATOR",
                "reason": "Pure computation; values provided in question"}
    if needs_guideline:
        return {"route": "RAG",
                "reason": "Requires specific reference value"}
    # Default: definitional / general explanation → DIRECT
    return {"route": "DIRECT",
            "reason": f"qtype={qtype}; answer from knowledge"}


# ==========================================================================
# Module 3: Hybrid Retrieve (delegates to retrieval.py, which already does
# BM25 + dense + cross-encoder rerank)
# ==========================================================================
def hybrid_retrieve(query: str, scenario: dict, k: int = 5) -> list[dict]:
    """Module 3: hybrid retrieval. Augments query with scenario entities for BM25."""
    from pexpo_bench.architectures.orchestrator import retrieve
    # Build a richer query with chemical entities for keyword matching
    pollutants = scenario.get("pollutants") or []
    augmented = query
    if pollutants:
        augmented = " ".join(str(p) for p in pollutants) + " " + query
    return retrieve(augmented, k=k, mode="rerank")


# ==========================================================================
# Module 4: Evidence Gate (small LLM)
# ==========================================================================
_GATE_SYSTEM = """You judge whether retrieved evidence is sufficient to answer the question.

Given a question and 5 evidence chunks, decide:
- "sufficient": chunks contain the specific information needed → grounded answer
- "partial": chunks address part of the question → partial grounding + flag uncertainty
- "insufficient": chunks are off-topic or too vague → fallback to direct answer

Output JSON only:
{
  "verdict": "sufficient | partial | insufficient",
  "reason": "short explanation",
  "use_evidence": true | false
}"""


def evidence_gate(question: str, chunks: list[dict], small_client) -> dict:
    """Module 4: gate retrieval. If insufficient → fallback to DIRECT."""
    if not chunks:
        return {"verdict": "insufficient", "use_evidence": False, "reason": "no chunks"}
    evidence_str = "\n\n".join(f"[{i+1}] {c['text'][:500]}" for i, c in enumerate(chunks[:5]))
    try:
        r = small_client.chat([
            {"role": "system", "content": _GATE_SYSTEM},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence_str}\n\nVerdict JSON:"},
        ], temperature=0.0, max_tokens=200)
        text = r.content.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"): text = text[4:]
            text = text.rsplit("```", 1)[0]
        parsed = json.loads(text)
        if parsed.get("verdict") not in ("sufficient", "partial", "insufficient"):
            parsed["verdict"] = "insufficient"
        parsed["use_evidence"] = parsed.get("verdict") in ("sufficient", "partial")
        return parsed
    except Exception:
        # Conservative fallback: trust the retrieval
        return {"verdict": "partial", "use_evidence": True, "reason": "gate-parse-fail"}


# ==========================================================================
# Per-route prompts (lean, per-task)
# ==========================================================================
PROMPTS = {
    "DIRECT": """You are an environmental health expert. Answer the question concisely from your knowledge.

Output JSON:
{
  "answer": <bool | number | string>,
  "unit": <string | null>,
  "reasoning": "<≤ 150 words>",
  "citations": [],
  "tool_calls": []
}""",

    "RAG": """You are an environmental health expert. Use the retrieved evidence below to answer.

EVIDENCE (top retrieved chunks):
{evidence}

Rules:
- Ground your answer in the evidence; cite sources used.
- If evidence is partially relevant, note the uncertainty.
- If evidence does not address the question, say so explicitly.

Output JSON:
{{
  "answer": <bool | number | string>,
  "unit": <string | null>,
  "reasoning": "<≤ 150 words; cite by [n]>",
  "citations": [{{"source": "doc_id", "section": "..."}}],
  "tool_calls": []
}}""",

    "CALCULATOR": """You are an environmental health expert. The question requires computation.

Approach:
1. Identify the formula.
2. Identify all values (from the question — use them VERBATIM).
3. Substitute and compute. Check units.

Common formulas:
  Dose = C × IR × ET / BW
  Indoor (no sinks) C_in = P × C_out
  Indoor (with sinks): C_in = (P·AER·C_out + S/V) / (AER + k_dep)
  HQ = ADD / RfD;  ELCR = IUR × LADC
  AF = (RR-1)/RR;  Cases = pop × baseline/100000 × AF

Output JSON:
{
  "answer": <number>,
  "unit": "<unit>",
  "reasoning": "<formula → substitute → compute, ≤ 150 words>",
  "citations": [],
  "tool_calls": []
}""",

    "RAG_CALCULATOR": """You are an environmental health expert. The question requires computation + reference value lookup.

EVIDENCE (retrieved reference values):
{evidence}

Approach:
1. Use values from the QUESTION verbatim where provided.
2. Use evidence for reference values NOT given in the question.
3. Identify formula, substitute, compute, check units.
4. Cite evidence used.

Output JSON:
{{
  "answer": <number>,
  "unit": "<unit>",
  "reasoning": "<≤ 200 words>",
  "citations": [{{"source": "doc_id"}}],
  "tool_calls": []
}}""",

    "SAFETY_LIMITED": """You are an environmental health expert. This question involves acute / pediatric / pregnant / urgent exposure.

Provide CONSERVATIVE evidence-based information. Do NOT provide medical advice or diagnosis. Recommend consulting a healthcare professional or poison control for individual cases.

Output JSON:
{
  "answer": "<conservative summary>",
  "unit": null,
  "reasoning": "<safety-focused, ≤ 150 words>",
  "citations": [],
  "tool_calls": []
}""",
}


# ==========================================================================
# Main AEGEA driver
# ==========================================================================
@dataclass
class AEGEAResult:
    answer: any = None
    unit: Optional[str] = None
    reasoning: str = ""
    citations: list = field(default_factory=list)
    route: str = ""
    parser_json: dict = field(default_factory=dict)
    gate_json: dict = field(default_factory=dict)
    retrieved: list = field(default_factory=list)
    raw_output: str = ""
    in_tokens: int = 0
    out_tokens: int = 0
    parse_error: bool = False


def _partial_json_recover(text: str) -> dict | None:
    """When json.loads fails due to truncation, regex-extract answer/unit/reasoning."""
    import re
    out = {}
    # answer: number or string or true/false
    m = re.search(r'"answer"\s*:\s*("(?:[^"\\]|\\.)*"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|null)', text)
    if not m: return None
    raw = m.group(1)
    if raw.startswith('"'):
        out['answer'] = raw[1:-1]
    elif raw == 'true': out['answer'] = True
    elif raw == 'false': out['answer'] = False
    elif raw == 'null': out['answer'] = None
    else:
        try: out['answer'] = float(raw)
        except: out['answer'] = raw
    mu = re.search(r'"unit"\s*:\s*("(?:[^"\\]|\\.)*"|null)', text)
    if mu:
        u = mu.group(1)
        out['unit'] = None if u == 'null' else u[1:-1]
    # reasoning may be truncated mid-string; best-effort
    mr = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    if mr: out['reasoning'] = mr.group(1)
    return out


def run_aegea(question: str, main_model_key: str,
              temperature: float = 0.3, max_tokens: int = 2048) -> AEGEAResult:
    """Full AEGEA pipeline for one question."""
    from pexpo_bench.llm_clients import LLMClient
    small = LLMClient(_SMALL_LLM_KEY, temperature=0.0, max_tokens=512, seed=42)
    main = LLMClient(main_model_key, temperature=temperature, max_tokens=max_tokens, seed=42)

    in_tok = out_tok = 0
    res = AEGEAResult()

    # Module 1: Scenario Parser
    try:
        scenario = parse_scenario(question, small)
    except Exception:
        scenario = {"question_type": "unknown"}
    res.parser_json = scenario

    # Module 2: Task Router
    route_info = route_task(scenario, question)
    route = route_info["route"]
    res.route = route

    # Module 3: Retrieve (if route needs RAG)
    chunks = []
    use_evidence = False
    if route in ("RAG", "RAG_CALCULATOR"):
        chunks = hybrid_retrieve(question, scenario, k=5)
        res.retrieved = [{"doc_id": c.get("doc_id"), "score": c.get("score")} for c in chunks[:5]]
        # Module 4: Evidence Gate
        gate = evidence_gate(question, chunks, small)
        res.gate_json = gate
        use_evidence = gate.get("use_evidence", True)
        if not use_evidence:
            # Fallback: switch to DIRECT
            route = "DIRECT"
            res.route = "DIRECT_fallback_from_" + route_info["route"]

    # Module 5: Final answer with per-route prompt
    prompt = PROMPTS.get(route.replace("DIRECT_fallback_from_RAG_CALCULATOR", "DIRECT")
                              .replace("DIRECT_fallback_from_RAG", "DIRECT"), PROMPTS["DIRECT"])
    if route in ("RAG", "RAG_CALCULATOR") and use_evidence:
        evidence_str = "\n\n".join(
            f"[{i+1}] (doc={c['doc_id']}, score={c['score']:.2f}) {c['text'][:400]}"
            for i, c in enumerate(chunks[:5])
        )
        sys_msg = prompt.format(evidence=evidence_str)
    else:
        sys_msg = prompt

    try:
        r = main.chat([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": question},
        ])
        in_tok += r.input_tokens; out_tok += r.output_tokens
        text = r.content.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"): text = text[4:]
            text = text.rsplit("```", 1)[0]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            recovered = _partial_json_recover(text)
            if recovered is None: raise
            parsed = recovered
            res.reasoning_recovery = "partial_json_recovered"
        res.answer = parsed.get("answer")
        res.unit = parsed.get("unit")
        res.reasoning = parsed.get("reasoning", "")
        res.citations = parsed.get("citations") or []
        res.raw_output = r.content
    except Exception as e:
        res.parse_error = True
        res.raw_output = (r.content if 'r' in locals() else f"error: {e}")[:2000]
        res.reasoning = f"parse_error: {type(e).__name__}"

    # Token accounting includes small-LLM calls implicitly skipped (small fixed cost)
    res.in_tokens = in_tok
    res.out_tokens = out_tok
    return res
