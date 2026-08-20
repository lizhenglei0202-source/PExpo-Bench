# PExpo-Bench Release

Benchmark, code, and complete experimental data for "A Defective Execution Environment
Can Manufacture LLM Architecture Effects: Corrected-Rerun Evidence from a 1,104-Item
Personal Exposure Assessment Benchmark".

## Contents
- `code/pexpo_bench/` — benchmark harness: five architectures (A0–A4 plus factorial
  variants), thread-safe tool sandbox, retrieval stack, runners, scoring, judges,
  analysis and figure scripts. Set `PEXPO_ROOT` to this release's root; put credentials
  in `.env` (see `.env.example` — no keys are shipped).
- `data/bank/` — the 1,104-item question bank with the per-item curation changelog and
  the curated 1,027-item evaluation set (361-item calculation stream separately).
- `data/trajectories/` — raw model trajectories: `main/` (4 models × 5 architectures),
  `factorial/` (8-corner harness decomposition, calculation stream), `seeds/` (three-seed
  replication).
- `data/scored/` — scored datasets (`results_main.parquet` is the paper's source of
  record).
- `data/judges/` — open-ended judgments, grounding adjudications, and the blinded
  judge-calibration sample with agreement analysis.
- `data/manifests/` — run manifests (bank checksums, endpoints, seeds) and the
  environment lock.
- `figures/`, `RESULTS_TABLES.md` — the paper's figures and numeric tables.
- `REPRODUCE.md` — end-to-end reproduction guide.

Knowledge-base source documents are not redistributed (copyright); the ingestion script
and document list allow rebuilding the retrieval index from public sources.
