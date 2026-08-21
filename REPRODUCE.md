# Reproduction guide

## 0. Prepare paths (once)
The analysis scripts address data by the layout used when the experiments were run;
this one-liner maps the published `data/` tree onto those paths with symlinks and
derives the programmatic-stream bank file from the full bank.

    bash setup_paths.sh
    export PEXPO_ROOT=$(pwd)

## 1. Rescore from released trajectories (no API calls)
    python code/pexpo_bench/analysis/build_scored_dataset.py

## 2. Regenerate tables and figures
    python code/pexpo_bench/analysis/make_tables.py
    python code/pexpo_bench/analysis/make_figures.py

## 3. Re-run the experiment (requires API credentials in .env)
    python -m pexpo_bench.runners.run_experiment \
      --bank data/bank/bank_evaluation_set.yaml --out runs/main --run-idx 1 \
      --models gpt-5.4 gpt-5.4-mini gpt-5.4-nano deepseek-v4 \
      --archs A0_naive A1_context_eng A2p_rag_constrained A3_agent A4p_hybrid_constrained

Before any run, validate the execution layer: the runner's smoke mode
(`--max-questions 2`) must show successful `python_sandbox` tool calls; per-tool error
rates are recorded in every trajectory. Model wire identifiers, endpoints, and bank
checksums are recorded in `data/manifests/`.
