# PExpo-Bench

Experimental code and data for **Tool Access Helps a Small Language Model Approach
Flagship Performance on a Personal Exposure Assessment Benchmark**.

## Experiment

The main experiment evaluates four models with five configurations on the
1,027-item question set. The accompanying analyses cover the 361-item calculation
factorial design, the tool-removal configuration A3-T, three seed replications,
reference grounding, expert validation and judge calibration.

| Configuration | Reference information | Execution |
|---|---|---|
| A0 | Base prompt | Direct response |
| A1 | Static reference context | Direct response |
| A2 | Static context and five retrieved passages, with evidence-use rules | Direct response |
| A3 | Tool instructions and reference lookups | 17 callable functions; eight rounds |
| A4 | Retrieval and tools, with evidence-use rules | 18 callable functions; ten rounds |
| A3-T | A3 prompt without tool instructions; reference lookups kept | 2 planning functions; eight rounds |

The factorial conditions use `F(R,P,B)`: retrieval availability, evidence-use rules,
and a ten-round budget. A3 and A4 supply the F000 and F111 end points.
See [CONFIGURATIONS.md](CONFIGURATIONS.md) for the complete design.

## Contents

- `code/pexpo_bench/`: model clients, configurations, prompts, retrieval, tools,
  experiment runners, the open-ended and reference-grounding judges, scoring and the
  analysis scripts that compute the reported tables.
- `data/bank/`: the 1,027-item evaluation set (`bank_evaluation_set.yaml`), the 361-item
  calculation stream used by the factorial design (`bank_calculation_stream.yaml`) and the
  545 true/false and calculation items used by the A0–A2 seed replication
  (`bank_objective_items.yaml`); the two smaller files are subsets of the evaluation set.
- `data/trajectories/`: every recorded model response: the main grid (20 cells × 1,027
  items), the six factorial arms (4 models × 361 items each), the tool-removal configuration
  (4 models × 1,027 items) and the seed replications (seeds 43–45; A3 and A4 on 1,017 items
  of the evaluation set, A0–A2 on the 545 objective items).
- `data/scored/`: the scoring tables (`results_main.parquet`, `results_all_phases.parquet`).
- `data/judges/`: open-ended judge scores for the main grid, the seed replications and the
  tool-removal configuration (one judge session) with an independent scoring pass of the
  main grid, reference-grounding judgments for all 20,540 main-grid responses, and the judge-calibration study with its human ratings.
- `data/expert_validation/`: the expert rating sheets, sampling manifest and agreement
  statistics.
- `data/manifests/`: the recorded run manifest and software environment.
- `RESULTS_TABLES.md`: the numerical results computed by the released scripts; the supplementary
  statistics, the tool-use split and the error-magnitude classification are written to
  `analysis_outputs/` by the scripts listed in REPRODUCE.md.
- `MANIFEST.json`: file sizes and SHA-256 checksums.
- `LICENSE`: MIT license for the software in this release.

Follow [REPRODUCE.md](REPRODUCE.md) to recompute the recorded results without model
API calls or to execute the experiments with your own credentials. Stored response
identifiers are resolved by `data_schema.py`; the command-line interface uses the
paper's A0–A4 and factorial labels.

The third-party document corpus and retrieval index require separate preparation.
The concentration lookup uses fixed defaults for uncached locations, and the
trajectory helper returns fixed example segments. These deterministic helper
behaviors are retained as used by the benchmark. Retrieved passages are distributed
as identifiers (`doc_id`, `section`, `chunk_id`, `score`), not as text
([SANITIZATION.md](SANITIZATION.md)).

Credentials are configured locally in `.env`. The public configuration example
contains blank values; private paths and workstation identifiers are redacted.
Manuscript figures and their source workbook accompany the journal submission.
