# Reproduction

Run commands from the extracted package root in an isolated Python environment.
The original environment record is retained in `data/manifests/environment_lock.txt`.
Use PyArrow 25.0.1 for the released Parquet files; PyArrow 19 cannot read them reliably.

```bash
export PEXPO_ROOT="$PWD"
export PYTHONPATH="$PEXPO_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install pandas numpy scipy PyYAML python-dotenv pyarrow==25.0.1
```

## Numerical reproduction without API calls

```bash
python -m pexpo_bench.analysis.build_scored_dataset
python -m pexpo_bench.analysis.make_tables
python -m pexpo_bench.analysis.grounding_stats
python -m pexpo_bench.analysis.trace_diagnostics
```

Outputs go to `analysis_outputs/`; input records in `data/` are not overwritten.
The main-table check compares 20,540 records (1,027 questions × 20 model/configuration
cells). Factorial and seed-replication results are included in the numerical summary.

## New model executions

Install the model/retrieval dependencies recorded in the original environment file
and place the required credentials in `.env` at this package root. The `.env.example`
file contains variable names only. Configure the model endpoints explicitly for the
intended provider snapshots; current defaults do not establish historical availability.

For retrieval configurations, supply an index directory through `PEXPO_INDEX_DIR`.
`Retriever.load` expects `chunks.parquet` and `faiss.index`, with the embedding and
reranking models specified in the implementation. The third-party corpus/index is
not included, so a new retrieval run requires this separately prepared resource.

```bash
python -m pexpo_bench.runners.run_experiment \
  --bank data/bank/bank_evaluation_set.yaml --out outputs/main --run-idx 1 \
  --models gpt-5.4 gpt-5.4-mini gpt-5.4-nano deepseek-v4 \
  --archs A0_naive A1_context_eng A2p_rag_constrained A3_agent A4p_hybrid_constrained \
  --seed 42 --temperature 0.3
```

`--seed`, `--run-idx`, checkpointing and transport retry behavior remain explicit.
To repeat the reported replication design use seeds 43, 44 and 45 with A3 and A4;
use separate output directories. Recorded failures must be included when scoring the
same evaluation set. Recorded provenance is retained to support reproduction.
