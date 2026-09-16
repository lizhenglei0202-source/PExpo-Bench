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
python -m pexpo_bench.analysis.trace_diagnostics
python data/judges/calibration/analyze_agreement.py --out analysis_outputs/judge_agreement.json
```

The main-table check recomputes 20,540 scores from response records and saved
open-ended judgments. Statistics include the factorial and seed-replication
datasets. Reference-grounding statistics use the fixed 299-item subsample.
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
for independent experiments and seed replications.

## Score new open-ended responses

```bash
python -m pexpo_bench.runners.run_open_judge \
  --runs outputs/main --bank data/bank/bank_evaluation_set.yaml \
  --out outputs/judgments/open_ended.jsonl
```

This step makes new cross-family judge calls. The distributed numerical analysis
uses the saved judgments and datasets in `data/`.
