# PExpo-Bench

Experimental code and data for **Tool Access Helps a Small Language Model Approach
Flagship Performance on a Personal Exposure Assessment Benchmark**.

## Experiment

The main experiment evaluates four models with five configurations on the curated
1,027-item question set. The accompanying analyses cover the 361-item calculation
factorial design, three seed replications, reference grounding, expert validation
and judge calibration.

| Configuration | Reference information | Execution |
|---|---|---|
| A0 | Base prompt | Direct response |
| A1 | Static reference context | Direct response |
| A2 | Static context and five retrieved passages, with evidence-use rules | Direct response |
| A3 | Tool instructions and reference lookups | 17 callable functions; eight rounds |
| A4 | Retrieval and tools, with evidence-use rules | 18 callable functions; ten rounds |

The factorial conditions use `F(R,P,B)`: retrieval availability, evidence-use rules,
and a ten-round budget. A3 and A4 supply the F000 and F111 end points.
See [CONFIGURATIONS.md](CONFIGURATIONS.md) for the complete design.

## Contents

- `code/pexpo_bench/`: model clients, configurations, prompts, retrieval, tools,
  experiment runners, scoring and statistical analysis.
- `data/bank/`: the 1,104-item full bank, 1,027-item evaluation set, calculation
  stream, curation records and the 2026-09-18 gold-standard revision records
  (`revision_20260918/`).
- `data/trajectories/`: main, factorial and seed-replication responses.
- `data/scored/`: the recorded main and combined scoring tables.
- `data/judges/`, `data/expert_validation/`: judgments, anonymous ratings and
  validation protocols.
- `data/manifests/`: execution settings and the recorded software environment.
- `RESULTS_TABLES.md`: numerical results.
- `MANIFEST.json`: file sizes and SHA-256 checksums.
- `LICENSE`: MIT license for the software in this release.

Follow [REPRODUCE.md](REPRODUCE.md) to reproduce the recorded results without model
API calls or to execute the specified experiments with your own credentials.
Stored response identifiers are resolved by `data_schema.py`; the command-line
interface uses the paper's A0–A4 and factorial labels.

The third-party document corpus and retrieval index require separate preparation.
The concentration lookup uses fixed defaults for uncached locations, and the
trajectory helper returns fixed example segments. These deterministic helper
behaviors are retained as used by the benchmark.

Credentials are configured locally in `.env`. The public configuration example
contains blank values; private paths and workstation identifiers are redacted.
Manuscript figures and their source workbook accompany the journal submission.

## Data version

This package carries the question bank and results after the gold-standard review of
18 September 2026: 97 questions were reworded and re-answered in every experiment,
87 answers or units were corrected and re-scored on the original responses, and 57
items received explanation-only edits; 1,304 of the 20,540 main-grid scores changed.
`data/bank/revision_20260918/` lists every change per item, and
`data/manifests/revision_20260918.yaml` records the execution scope. The
reference-grounding judgments were re-run for the 16 reworded subsample items; the seed-replication files contain
the 545 objective items used in the analysis. Retrieved passages are distributed as
identifiers (`doc_id`, `section`, `chunk_id`, `score`), not as text (SANITIZATION.md).
