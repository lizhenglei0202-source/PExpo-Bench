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

## Seed replications

A3 and A4 are evaluated on the 1,027-item bank for all four models with requested
seeds 43, 44 and 45. Statistical comparisons of seed effects use the 545
true/false and calculation items, as specified in the manuscript.
The distributed seed-replication files contain those 545 items; open-ended seed
responses were not re-generated after the 2026-09-18 revision and are not distributed.

## Evaluation

- True/false and calculation scoring: `analysis/scoring.py`.
- Open-ended rubric and runner: `runners/run_open_judge.py`.
- Open-ended judges: DeepSeek-V4 for GPT outputs; GPT-5.4-nano for DeepSeek outputs.
- Claim-extraction/entailment judges: DeepSeek-V4 for GPT outputs; GPT-4o-mini for
  DeepSeek outputs.
- Judge calibration: `data/judges/calibration/`.
- Statistical comparisons: `analysis/make_tables.py`.
- Reference-grounding summary: `analysis/grounding_stats.py`.
- Tool definitions supplied to models: `TOOL_DEFS` in
  `architectures/orchestrator.py`.

Prompt text, function schemas and execution budgets are defined by the
configuration classes. The recorded data retain their original identifiers;
`data_schema.py` maps these identifiers to the labels above.
