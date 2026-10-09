# Experiment configurations

## Main experiment

- Models: GPT-5.4, GPT-5.4-mini, GPT-5.4-nano and DeepSeek-V4.
- Configurations: A0, A1, A2, A3 and A4.
- Bank: `data/bank/bank_evaluation_set.yaml`, 1,027 items.
- Temperature: 0.3; requested seed: 42; run identifier: 1.
- Outputs: `outputs/main/<model>/<configuration>/run_1.jsonl`.

Provider-specific request handling is defined in `llm_clients.py`. The seed is
sent to endpoints that accept it under this protocol; the native OpenAI calls
omit the seed argument. Identical requested seeds do not guarantee identical
responses across endpoints.

## Factorial experiment

R enables retrieval; P enables evidence-use rules; B sets the execution budget
to ten rounds instead of eight.

| Configuration | R | P | B | Round limit |
|---|---:|---:|---:|---:|
| A3 (F000) | 0 | 0 | 0 | 8 |
| F100 | 1 | 0 | 0 | 8 |
| F010 | 0 | 1 | 0 | 8 |
| F001 | 0 | 0 | 1 | 10 |
| F110 | 1 | 1 | 0 | 8 |
| F101 | 1 | 0 | 1 | 10 |
| F011 | 0 | 1 | 1 | 10 |
| A4 (F111) | 1 | 1 | 1 | 10 |

The six additional conditions use the 361-item calculation stream, all four
models, temperature 0.3 and seed 42. A3/A4 calculation responses come from the
main experiment. Retrieval prompts retain the recorded ten-round wording even
in the B=0 conditions; the execution loop enforces the eight-round limit there.

## Tool-removal configuration

A3-T keeps the A3 system prompt without its tool list and tool-use guidance, the reference
data, the citable-source list, the plan-first flow, the eight-round limit, the decoding
settings and the FINAL output format, and declares only the two planning functions
(`submit_plan`, `revise_plan`); a call to any other function is refused and returned to the
model as an error message, and the planning acknowledgement reads "proceed to FINAL"
(`prompts._NOTOOLS_EDITS` lists every difference between the two prompts). All four models,
the 1,027-item bank, temperature 0.3, requested seed 42. Outputs:
`outputs/tool_removal/<model>/A3-T/run_1.jsonl`; recorded data key `A3n_agent_notools`.
A3 − A3-T estimates the effect of tool access and A3-T − A0 the effect of the prompt and
planning scaffold (`analysis/make_tables.py`, phase D of `results_all_phases.parquet`).

## Seed replications

A3 and A4 are evaluated on the 1,027-item bank for all four models with requested
seeds 43, 44 and 45; A0, A1 and A2 are evaluated on the 545 true/false and
calculation items (`data/bank/bank_objective_items.yaml`) with the same seeds.
The distributed A3 and A4 repeat files contain the 545 objective items and 472 of the
482 open-ended items; the ten open-ended items whose wording in the released evaluation
set differs from the wording the repeats were run with are not distributed. Seed
statistics are reported on the 545 objective items for every configuration and, for A3
and A4, on the 1,017-item union; the open-ended repeat responses were scored by the
cross-family judges in the same judge session as the main grid
(`data/judges/open_ended_judgments_seeds.jsonl`).

## DeepSeek-V4 decoding

DeepSeek-V4 (`deepseek-v4-flash`) is called in the provider's thinking mode in every
subject run: hidden reasoning tokens are generated before the answer and count against
the 2,048-token completion budget, so `output_tokens` in the trajectories exceeds the
visible answer length. A response whose visible content is empty when the budget is
exhausted is scored 0 in every configuration (`analysis/scoring.py`,
`analysis/build_scored_dataset.py`). Judge calls to DeepSeek-V4 (open-ended
scoring, claim extraction and entailment) disable thinking. `DEEPSEEK_THINKING=disabled`
switches subject runs to the non-thinking mode for diagnostics only.

## Evaluation

- True/false and calculation scoring: `analysis/scoring.py`.
  Calculation answers of the FINAL-line configurations (A3, A4 and the factorial arms) are read
  from the recorded FINAL text with a parser that accepts scientific notation; the execution layer's
  `answer` field keeps only the mantissa of such notation and is not used for scoring.
- Reference values: the inhalation defaults carried by the primer and by `exposure_factor_lookup`
  (adult 15.7, adult-male 16.0, adult-female 12.0 m³/day) are, respectively, the EFH 2011 Table 6-1
  long-term means for 21 to <31 y and 51 to <61 y, the 31 to <51 y mean, and a benchmark default;
  Table 6-1 reports males and females combined. The values are kept as the recorded runs used them.
- Open-ended rubric and runner: `runners/run_open_judge.py`.
- Open-ended judges: DeepSeek-V4 for GPT outputs; GPT-5.4-nano for DeepSeek outputs. Every
  reported open-ended score (main grid, seed replications, tool-removal configuration) comes
  from one judge session: `data/judges/open_ended_judgments.jsonl`,
  `open_ended_judgments_seeds.jsonl` and `open_ended_judgments_tool_removal.jsonl`.
  `open_ended_judgments_prior_pass.jsonl` holds an independent scoring pass of the main-grid
  answers by the same judge models on another deployment of the DeepSeek-V4 judge;
  `make_tables` reports the agreement between the two passes (judge reproducibility).
- Claim-extraction/entailment judges: DeepSeek-V4 for GPT outputs; GPT-4o-mini for
  DeepSeek outputs.
- Judge calibration: `data/judges/calibration/`.
- Statistical comparisons: `analysis/make_tables.py`.
- Reference-grounding judge runner: `runners/run_hr_judge.py` (claim extraction and
  entailment in `evaluation/hr_atomic_judge.py`); summary: `analysis/grounding_stats.py`.
- Tool definitions supplied to models: `TOOL_DEFS` in
  `architectures/orchestrator.py`.

Prompt text, function schemas and execution budgets are defined by the
configuration classes. The recorded data retain their original identifiers;
`data_schema.py` maps these identifiers to the labels above.
