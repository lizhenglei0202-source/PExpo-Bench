# Judge Calibration Package — PEXPO-Bench (built 2026-08-12)

Purpose: validate the automated LLM judge used for open-ended scoring by
(a) double-judging a stratified sample with two judge models and
(b) collecting blinded human ratings on the same sample.

## Sample

- Source: `runs/v3_scored/all_scored_v2.parquet`, ACTIVE (`retired == False`)
  `open_ended` items only — 482 unique qids.
- 100 qids stratified-sampled by subdomain x difficulty (15 non-empty strata),
  proportional allocation with at least 1 per stratum, largest-remainder
  rounding, numpy `default_rng(42)`. Full allocation in `sample_manifest.json`.
- For each sampled qid, the `A3_agent` answer of all four models
  (gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, deepseek-v4) was pulled from
  `runs/v3_main/<model>/A3_agent/run_1.jsonl` → 400 (item, answer) rows.
- Question text and gold answer/rationale come from
  `pexpo_bench/samples/pexpo_bench_v3_full.patched_20260811.yaml`.
  Gold and predicted answers are truncated to 2500 chars, exactly as in
  `run_open_judge.py`.

## Blinding

Rows were shuffled with `default_rng(42)` and assigned codes JC-001..JC-400.
`human_rating_sheet.csv` contains NO model identity. The mapping
code → (qid, model) lives in `blinding_key.json` — **withhold this file (and
`judge_inputs.jsonl`) from raters until both rating columns are complete.**

## Files

| File | Role |
|---|---|
| `sample_manifest.json` | sampled qids, strata, allocation, seed |
| `human_rating_sheet.csv` | 400 blinded rows; COMPLETED 2026-08-21 (both rater columns filled) |
| `INSTRUCTIONS.md` | rater instructions with the exact 0-5 rubric from `run_open_judge.py` |
| `blinding_key.json` | code → model/qid map — WITHHELD from raters |
| `judge_inputs.jsonl` | machine-readable rows for the double-judge run (contains model identity) |
| `run_double_judge.py` | scores all 400 rows with BOTH judges — EXECUTED 2026-08-13; output in `per_row_double_judge.jsonl` |
| `analyze_agreement.py` | agreement statistics (no API calls) |
| `_cost_estimate.json` | machine-readable copy of the cost estimate below |

## Status (2026-08-13)

The machine-vs-machine half is **complete**: 800 judgments across 400 answers, both
judges. Result: Pearson r = 0.73, within-one-point agreement 90%, quadratic κ = 0.58,
and the GPT-5.4-nano judge is +0.38 points more lenient on the 0–5 scale (reported in
SI Section C). The **human** half is now complete as well (2026-08-21): both raters scored all 400
rows. Rater-to-rater agreement r = 0.92 (quadratic κ = 0.92, within-one 99.5%); against
the mean human score the DeepSeek-V4 judge is −0.15 and the GPT-5.4-nano judge +0.24,
so the human standard sits between the two judges. Full statistics in
`agreement_report.json`; the completed sheet is preserved as
`human_rating_sheet_completed_20260821.csv`.

## Workflow

1. Give raters `human_rating_sheet.csv` + `INSTRUCTIONS.md` only. Two raters
   score independently.
2. Run the double judge (makes API calls; ALREADY DONE — results shipped):
   `python run_double_judge.py` → `per_row_double_judge.jsonl`
   (800 judgments: 400 rows x 2 judges — gpt-5.4-nano and deepseek-v4 —
   identical prompt/rubric/settings to the production judge: temperature 0.0,
   seed 42, max_tokens 16 for nano / 600 for deepseek-v4; resumable).
3. When the human columns are filled, run
   `python analyze_agreement.py` → `agreement_report.json` with per-judge
   means (overall and by source model), judge-judge Pearson/Spearman,
   exact/within-1 agreement, unweighted/linear/quadratic Cohen's kappa,
   rater1-rater2 agreement, and each judge vs the mean human score.

## API cost of the double-judge run (spent 2026-08-13)

Computed from actual character counts of the 400 assembled prompts
(question + gold + answer + ~700-char template), at ~4 chars/token:

- Input: ~192k tokens per judge (~384k total).
- Output: gpt-5.4-nano emits a single integer (~5 tok/row → ~2k tok);
  deepseek-v4 is a reasoning model capped at 600 tok (~550 tok/row → ~220k tok).

| Judge | Input tok | Output tok | Assumed $/MTok (in/out) | Est. cost |
|---|---|---|---|---|
| gpt-5.4-nano | ~192k | ~2k | 0.05 / 0.40 | ~$0.01 |
| deepseek-v4 | ~192k | ~220k | 0.28 / 1.10 | ~$0.30 |
| **Total** | | | | **~$0.31** (order of magnitude: well under $1) |

Prices are assumptions recorded in `_cost_estimate.json`; rescale linearly if
actual list prices differ. Wall-clock at concurrency 8: roughly 10-20 min,
dominated by deepseek-v4 reasoning latency.
