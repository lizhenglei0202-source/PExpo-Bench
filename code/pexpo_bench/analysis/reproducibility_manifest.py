"""Reproducibility manifest auto-generator.

For every run, dump a YAML manifest containing:
  • exact model snapshot strings (per MODEL_REGISTRY)
  • prompt SHA256 hashes (per architecture)
  • KB content hash (chunks.parquet)
  • bank content hash
  • tool registry signature (tool names + arg schemas)
  • seed / temperature / concurrency
  • git commit (if a repo)
  • timestamp / wall-clock start

This file goes into paper SI; with it, anyone can replay the experiment.
"""
from __future__ import annotations
import os

import hashlib
import json
import pathlib
import subprocess
import sys
import time

import yaml


def _file_sha256(path: str | pathlib.Path) -> str:
    p = pathlib.Path(path)
    if not p.exists(): return "MISSING: [REDACTED_PATH]"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(pathlib.Path(__file__).resolve().parents[3]),
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        return out
    except Exception:
        return None


def _tool_registry_signature() -> dict:
    """Hash the tool registry shape (tool names + arg schemas)."""
    from pexpo_bench.architectures.orchestrator import A4_Hybrid
    sig = {td["function"]["name"]: td["function"]["parameters"]
           for td in A4_Hybrid.TOOL_DEFS}
    return {
        "n_tools": len(sig),
        "tool_names_sorted": sorted(sig.keys()),
        "schema_hash": _text_sha256(json.dumps(sig, sort_keys=True, default=str)),
    }


def _prompt_hashes() -> dict:
    from pexpo_bench.architectures import prompts as pmod
    keys = ["A0_SYSTEM", "A1_SYSTEM", "A2_SYSTEM", "A2P_SYSTEM",
            "A3_SYSTEM", "A4_SYSTEM", "A4P_SYSTEM"]
    return {k: _text_sha256(getattr(pmod, k, "")) for k in keys}


def _model_snapshots(model_keys: list[str]) -> dict:
    from pexpo_bench.llm_clients import MODEL_REGISTRY
    snaps = {}
    for k in model_keys:
        cfg = MODEL_REGISTRY.get(k, {})
        snaps[k] = {
            "model_string": cfg.get("model"),
            "base_url": "[REDACTED_ENDPOINT]",
            "price_in_usd_per_1M": cfg.get("price_in"),
            "price_out_usd_per_1M": cfg.get("price_out"),
        }
    return snaps


def write_manifest(
    out_dir: pathlib.Path, run_idx: int,
    bank_path: str | pathlib.Path,
    models: list[str], archs: list[str],
    temperature: float, seed: int, concurrency: int,
    kb_chunks_path: str | None = None,
) -> pathlib.Path:
    """Dump a YAML manifest. Returns the path."""
    from pexpo_bench.evaluation.judge_dispatch import JUDGE_FOR

    if kb_chunks_path is None:
        index_dir = pathlib.Path(os.environ.get("PEXPO_INDEX_DIR", str(pathlib.Path(__file__).resolve().parents[1] / "knowledge_base/index")))
        kb_chunks_path = str(index_dir / "chunks.parquet")
    manifest = {
        "run_idx": run_idx,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": _safe_hostname(),
        "git_commit_pexpo_bench": _git_commit(),
        "python_version": sys.version.split()[0],
        # Code / config hashes
        "bank_path": "[REDACTED_PATH]",
        "bank_sha256_16": _file_sha256(bank_path),
        "kb_chunks_path": "[REDACTED_PATH]",
        "kb_chunks_sha256_16": _file_sha256(kb_chunks_path),
        "prompt_hashes": _prompt_hashes(),
        "tool_registry": _tool_registry_signature(),
        # Experiment knobs
        "models": models,
        "model_snapshots": _model_snapshots(models),
        "architectures": archs,
        "temperature": temperature,
        "seed": seed,
        "concurrency": concurrency,
        # Judge dispatch
        "judge_dispatch": JUDGE_FOR,
        "judge_models_used": _model_snapshots(list(set(JUDGE_FOR.values()))),
        "notes": [
            "Run identifiers, seeds, model snapshots and retrieval-index hashes identify this execution.",
            "Replications and factorial settings must be recorded explicitly in the run arguments.",
        ],
    }
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"manifest_run_{run_idx}.yaml"
    path.write_text(yaml.dump(manifest, allow_unicode=True,
                              default_flow_style=False, sort_keys=False))
    return path


def _safe_hostname() -> str:
    try:
        import socket
        return "[REDACTED_HOST]"
    except Exception:
        return "unknown"


