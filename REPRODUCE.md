# Reproduction

Run commands from the extracted package root in an isolated Python environment.

```bash
export PEXPO_ROOT="$PWD"
export PYTHONPATH="$PEXPO_ROOT/code"
python -m pip install pandas numpy scipy PyYAML python-dotenv pyarrow==25.0.1
```

The recorded software environment is listed in
`data/manifests/environment_lock.txt`. PyArrow 25.0.1 is required to read the
distributed Parquet files reliably.

## Recompute recorded results

These commands make no model API calls and write to `analysis_outputs/`.

```bash
python -m pexpo_bench.analysis.build_scored_dataset
python -m pexpo_bench.analysis.make_tables
python -m pexpo_bench.analysis.grounding_stats
python -m pexpo_bench.analysis.supplementary_statistics
python -m pexpo_bench.analysis.trace_diagnostics
python -m pexpo_bench.analysis.tool_use_split
python -m pexpo_bench.analysis.error_magnitude
python data/judges/calibration/analyze_agreement.py --out analysis_outputs/judge_agreement.json
```

The main-table check recomputes 20,540 scores from response records and saved
open-ended judgments. Calculation answers of the FINAL-line configurations (A3, A4 and the
factorial arms) are read from the recorded FINAL text by `analysis/scoring.py`, whose numeric
parser accepts scientific notation written with a multiplication sign or a superscript
exponent; the `answer` field of those records keeps the execution layer's promotion. Statistics include the factorial, tool-removal and
seed-replication datasets (seeds 43–45: A0–A2 on the 545 objective items; A3/A4 on those
items and on 472 open-ended items), the judge-reproducibility table and the tool-removal
contrasts (A3 − A3-T, A3-T − A0). Reference-grounding statistics use all 20,540 main-grid rows.
Input data files are preserved.

## Execute experiments

Install the model and retrieval dependencies listed in the environment record.
Copy `.env.example` to a local `.env` and configure your model credentials and
endpoints. New model calls incur provider charges.

Retrieval conditions require `PEXPO_INDEX_DIR` containing `chunks.parquet` and
`faiss.index`, together with the embedding and reranking models specified in
`architectures/orchestrator.py`. Supply the external corpus and index before
running these conditions.

Inspect the commands with `--dry-run`, then remove that option to execute:

```bash
python -m pexpo_bench.runners.run_paper --experiment main --dry-run
python -m pexpo_bench.runners.run_paper --experiment factorial --dry-run
python -m pexpo_bench.runners.run_paper --experiment tool_removal --dry-run
python -m pexpo_bench.runners.run_paper --experiment seeds --dry-run
```

Use `--models` to select a subset of the four evaluated models and `--out` to set
the output directory. The complete settings are in
[CONFIGURATIONS.md](CONFIGURATIONS.md).

For individual configurations:

```bash
python -m pexpo_bench.runners.run_experiment \
  --bank data/bank/bank_evaluation_set.yaml --out outputs/main \
  --models gpt-5.4 gpt-5.4-mini gpt-5.4-nano deepseek-v4 \
  --archs A0 A1 A2 A3 A4 --seed 42 --temperature 0.3 --run-idx 1
```

Checkpointing preserves completed responses. Transient transport retries and
recorded model failures remain explicit in the output. Use separate directories
for independent experiments and seed replications. DeepSeek-V4 subject runs use
the provider's thinking mode, as the recorded runs did (CONFIGURATIONS.md).

## Score new responses

Open-ended answers:

```bash
python -m pexpo_bench.runners.run_open_judge \
  --runs outputs/main --bank data/bank/bank_evaluation_set.yaml \
  --out outputs/judgments/open_ended.jsonl
```

Reference grounding (claim extraction and entailment against the knowledge-base index):

```bash
python -m pexpo_bench.runners.run_hr_judge --runs outputs/main --out outputs/judgments/grounding
```

Both steps make new cross-family judge calls with thinking disabled for the DeepSeek-V4
judge; failed judge calls are reported per row and re-run on resume rather than cached.
The distributed numerical analysis uses the saved judgments and datasets in `data/`.

## Supplementary statistics

`pexpo_bench.analysis.supplementary_statistics` computes the quantities specified in
Methods Section 2.6 that `make_tables` does not emit: per-cell 95% confidence intervals
from 500 resamples, Kendall rank correlations between configuration orderings, Pareto
frontier membership under the dated price scenario on the full evaluation set and on the
545 objective items, the six cross-cell contrasts against GPT-5.4 A0 with Holm correction
as one family, the headline cross-cell contrasts (GPT-5.4-mini A3, the recommended
configuration, and the same model's A4 cell, each against GPT-5.4 A3 and GPT-5.4 A4), and the
two one-sided equivalence tests at margins of 2 and 1.5 percentage points. It writes
`analysis_outputs/supplementary_statistics.json` and a readable `.md` beside it.

`pexpo_bench.analysis.tool_use_split` splits the A3−A0 gain of every base model by whether the
recorded A3 trace invoked an external tool (plan meta-tools excluded): no external tool,
python_sandbox only, or a domain lookup or calculation function. It writes
`analysis_outputs/tool_use_split.json` and `.md` (Table S11).

`pexpo_bench.analysis.error_magnitude` classifies every main-grid calculation response by
the size of its error (correct, near miss, wrong by <2×, 2–10× or ≥10×, no extractable number),
reports the share of ≥10× errors that are exact decimal-prefix slips, and writes
`analysis_outputs/error_magnitude.json` and `.md` (SI Section K, Table S12, Figure S4).
