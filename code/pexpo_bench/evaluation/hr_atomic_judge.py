"""HR_reasoning — atomic claim + KB entailment.

Pipeline per (model, arch, qid):
  1. Extract atomic claims from model.reasoning  (LLM, cross-family judge)
  2. For each claim, FAISS-search whole KB (with dedup + low-info filter)
  3. Ask judge: "Does any retrieved chunk support this claim?" (entailment)
  4. HR_reasoning = unsupported / total

No doc_id is used anywhere.  Claims are decomposed at the SENTENCE level
preserving scope/units/conditionals (unlike RefChecker's strict s-p-o triples
which lose these).

Cache: per-question claim extraction is cached by (model_key, qid, reasoning_hash)
so re-running doesn't re-pay.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional

from pexpo_bench.llm_clients import LLMClient
import threading as _threading

# The two JSONL caches are read into memory once per path instead of being re-read line by
# line on every lookup. Lookup semantics are unchanged: the first record written for a key is
# the one returned.
_CACHE_LOCK = _threading.Lock()
_CACHES: dict[str, dict] = {}


def _cache_table(cache_path: pathlib.Path) -> dict:
    key = str(cache_path)
    with _CACHE_LOCK:
        tbl = _CACHES.get(key)
        if tbl is None:
            tbl = {}
            if cache_path.exists():
                for line in cache_path.read_text().splitlines():
                    if not line.strip(): continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    k = rec.get("key")
                    if k is not None and k not in tbl:
                        tbl[k] = rec
            _CACHES[key] = tbl
        return tbl


def _cache_put(cache_path: pathlib.Path, rec: dict) -> None:
    tbl = _cache_table(cache_path)
    with _CACHE_LOCK:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        tbl.setdefault(rec["key"], rec)


# ==========================================================================
# Claim extraction prompt
# ==========================================================================
_CLAIM_EXTRACTION_SYSTEM = """You decompose scientific reasoning into atomic factual claims.

A claim is ONE assertable proposition. Preserve scope, units, conditionals, and quantifiers.

GOOD examples of atomic claims:
  ✓ "Adult mean inhalation rate per EPA EFH Table 6-1 is 15.7 m³/day."
  ✓ "WHO AQG 2021 annual PM2.5 limit was tightened from 10 to 5 µg/m³."
  ✓ "Indoor PM2.5 concentration with no sinks equals P × outdoor concentration."

BAD examples (too coarse / too fine):
  ✗ "PM2.5 is bad."  (vague, not falsifiable)
  ✗ "EPA"  (not a claim)
  ✗ "Adults breathe."  (too generic; drop)

Output: a JSON array of strings (the claims). Do not output explanations.
If the reasoning is < 30 words or contains no factual claims, output [].
"""

_CLAIM_EXTRACTION_USER_TEMPLATE = """REASONING:
{reasoning}

QUESTION CONTEXT (for understanding scope):
{question}

Extract atomic claims. Return ONLY a JSON array of strings."""


# ==========================================================================
# Entailment prompt
# ==========================================================================
_ENTAILMENT_SYSTEM = """You judge whether scientific evidence supports a claim.

Given a CLAIM and 5 EVIDENCE CHUNKS retrieved from a knowledge base, decide:

  • "supported"      — at least one chunk's content entails or directly supports the claim
  • "contradicted"   — at least one chunk directly contradicts the claim
  • "no_info"        — none of the chunks address the claim either way

Output JSON: {"label": "supported" | "contradicted" | "no_info", "reason": "<≤25 words>"}

Be strict: surface keyword overlap is NOT support unless the chunk's meaning
actually entails the claim. Pay attention to units, quantifiers, scope.
"""

_ENTAILMENT_USER_TEMPLATE = """CLAIM:
{claim}

EVIDENCE CHUNKS (top-5 from KB):
{evidence}

Verdict in JSON only."""


# ==========================================================================
# Data classes
# ==========================================================================
@dataclass
class ClaimJudgment:
    qid: str
    arch: str
    model: str
    judge_model: str
    claim: str
    label: str               # supported | contradicted | no_info
    reason: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    judge_in_tok: int = 0
    judge_out_tok: int = 0
    judge_latency_s: float = 0.0


@dataclass
class HRResult:
    qid: str
    arch: str
    model: str
    judge_model: str
    n_claims: int
    n_supported: int
    n_contradicted: int
    n_no_info: int
    HR_reasoning: float | None    # NaN-like if n_claims < 2
    judgments: list[ClaimJudgment] = field(default_factory=list)
    extraction_failed: bool = False   # extractor call failed (row must be re-run)
    n_judge_failed: int = 0           # entailment calls that failed (row must be re-run)


# ==========================================================================
# Core
# ==========================================================================
def _hash_reasoning(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:12]


def extract_claims(reasoning: str, question: str,
                   judge_client: LLMClient,
                   cache_path: Optional[pathlib.Path] = None,
                   qid: str = "", subject_model: str = "") -> tuple[list[str], dict]:
    """Run claim extraction. Returns (claims, usage_meta).

    If cache_path given, looks up by hash before calling LLM.
    """
    rh = _hash_reasoning(reasoning)
    cache_key = f"{subject_model}|{qid}|{rh}"
    # Cache hit
    if cache_path:
        rec = _cache_table(cache_path).get(cache_key)
        if rec is not None:
            return rec["claims"], {"cache_hit": True, "call_failed": False}

    # Call judge
    t0 = time.time()
    resp = judge_client.chat([
        {"role": "system", "content": _CLAIM_EXTRACTION_SYSTEM},
        {"role": "user", "content": _CLAIM_EXTRACTION_USER_TEMPLATE.format(
            reasoning=(reasoning or "")[:3000],
            question=(question or "")[:500])},
    ], temperature=0.0, max_tokens=1024)
    raw = resp.content.strip()
    # Strip code fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    try:
        claims = json.loads(raw)
        if not isinstance(claims, list):
            claims = []
        claims = [str(c).strip() for c in claims if isinstance(c, (str, int, float))][:20]
    except Exception:
        claims = []
    usage = {
        "in_tok": resp.input_tokens, "out_tok": resp.output_tokens,
        "latency_s": time.time() - t0, "cache_hit": False,
    }
    # Write to cache — but NEVER cache the empty result of a failed call
    # Empty responses and error finish reasons must remain retryable.
    call_failed = (resp.finish_reason or "").startswith("error:") or not resp.content.strip()
    usage["call_failed"] = bool(call_failed)
    if cache_path and not call_failed:
        _cache_put(cache_path, {"key": cache_key, "claims": claims,
                                "subject_model": subject_model, "qid": qid})
    return claims, usage


def judge_claim(claim: str, evidence_chunks: list[str],
                judge_client: LLMClient,
                cache_path: Optional[pathlib.Path] = None) -> tuple[str, str, dict]:
    """Judge one claim against 5 chunks. Returns (label, reason, usage_meta)."""
    if not evidence_chunks:
        return "no_info", "no evidence retrieved", {"in_tok": 0, "out_tok": 0, "latency_s": 0}

    cache_key = hashlib.md5(
        (claim + "||" + "||".join(c[:200] for c in evidence_chunks)).encode("utf-8")
    ).hexdigest()[:16]
    if cache_path:
        rec = _cache_table(cache_path).get(cache_key)
        if rec is not None:
            return rec["label"], rec.get("reason", ""), {"cache_hit": True, "call_failed": False}

    evidence_str = "\n\n".join(f"[{i+1}] {c[:600]}" for i, c in enumerate(evidence_chunks[:5]))
    t0 = time.time()
    resp = judge_client.chat([
        {"role": "system", "content": _ENTAILMENT_SYSTEM},
        {"role": "user", "content": _ENTAILMENT_USER_TEMPLATE.format(
            claim=claim[:500], evidence=evidence_str)},
    ], temperature=0.0, max_tokens=200)
    raw = resp.content.strip()
    # An API error swallowed by the client (finish_reason "error:<Name>") or an empty
    # completion is not a verdict. It is reported as a failed call and never cached, so a
    # resumed run re-judges the claim instead of carrying an empty NO_INFO forward.
    call_failed = (resp.finish_reason or "").startswith("error:") or not raw
    if call_failed:
        return "no_info", "", {"in_tok": resp.input_tokens, "out_tok": resp.output_tokens,
                               "latency_s": time.time() - t0, "cache_hit": False, "call_failed": True}
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    try:
        parsed = json.loads(raw)
        label = str(parsed.get("label", "no_info")).lower()
        reason = str(parsed.get("reason", ""))[:200]
    except Exception:
        # Fallback: if model returned non-JSON, parse keywords
        low = raw.lower()
        if "supported" in low and "not" not in low: label = "supported"
        elif "contradict" in low: label = "contradicted"
        else: label = "no_info"
        reason = raw[:200]
    if label not in ("supported", "contradicted", "no_info"):
        label = "no_info"
    usage = {"in_tok": resp.input_tokens, "out_tok": resp.output_tokens,
             "latency_s": time.time() - t0, "cache_hit": False, "call_failed": False}
    if cache_path:
        _cache_put(cache_path, {"key": cache_key, "label": label, "reason": reason})
    return label, reason, usage


def evaluate_hr_reasoning(
    result: dict,                       # per-question model output (jsonl row)
    retriever,                          # pexpo_bench.retrieval.Retriever (with filters)
    judge_client: LLMClient,
    judge_model_key: str,
    cache_dir: pathlib.Path | None = None,
    k_evidence: int = 5,
) -> HRResult:
    """Full pipeline for one (qid, arch, model) record.

    Returns HR_reasoning ∈ [0, 1] with breakdown.
    Skipping rule: if fewer than 2 atomic claims extracted → HR = NaN-like (None)
                   to avoid 'rewarded for saying nothing'.
    """
    qid = result.get("qid", "")
    arch = result.get("architecture", "")
    subject_model = result.get("_model_key") or result.get("model", "")
    reasoning = result.get("reasoning", "") or ""
    question = result.get("_question_text") or result.get("question", "")

    claim_cache = (cache_dir / "claim_extraction_cache.jsonl") if cache_dir else None
    entail_cache = (cache_dir / "entailment_cache.jsonl") if cache_dir else None

    claims, ext_usage = extract_claims(reasoning, question, judge_client,
                                       cache_path=claim_cache, qid=qid,
                                       subject_model=subject_model)
    n = len(claims)
    if n < 2:
        return HRResult(qid=qid, arch=arch, model=subject_model,
                        judge_model=judge_model_key,
                        n_claims=n, n_supported=0,
                        n_contradicted=0, n_no_info=0,
                        HR_reasoning=None, judgments=[],
                        extraction_failed=bool(ext_usage.get("call_failed")))
    n_failed = 0

    judgments = []
    n_sup = n_con = n_no = 0
    for claim in claims:
        # KB retrieve (uses dedup + low-info filter)
        try:
            passages = retriever.retrieve(claim, k=k_evidence, mode="hybrid")
            chunk_texts = [p.text for p in passages]
            chunk_ids = [p.chunk_id for p in passages]
        except Exception:
            chunk_texts, chunk_ids = [], []
        label, reason, judge_usage = judge_claim(claim, chunk_texts,
                                                  judge_client, entail_cache)
        if judge_usage.get("call_failed"): n_failed += 1
        if label == "supported": n_sup += 1
        elif label == "contradicted": n_con += 1
        else: n_no += 1
        judgments.append(ClaimJudgment(
            qid=qid, arch=arch, model=subject_model,
            judge_model=judge_model_key, claim=claim, label=label,
            reason=reason, retrieved_chunk_ids=chunk_ids,
            judge_in_tok=judge_usage.get("in_tok", 0),
            judge_out_tok=judge_usage.get("out_tok", 0),
            judge_latency_s=judge_usage.get("latency_s", 0),
        ))

    # HR = (unsupported = contradicted + no_info) / total
    unsupported = n_con + n_no
    hr = unsupported / n if n else None
    return HRResult(
        qid=qid, arch=arch, model=subject_model,
        judge_model=judge_model_key,
        n_claims=n, n_supported=n_sup,
        n_contradicted=n_con, n_no_info=n_no,
        HR_reasoning=hr, judgments=judgments,
        n_judge_failed=n_failed,
    )
