#!/usr/bin/env bash
# Map the published data/ layout onto the paths the analysis scripts expect.
# Run once from the package root, then follow REPRODUCE.md:
#     bash setup_paths.sh && export PEXPO_ROOT=$(pwd)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

mkdir -p pexpo_bench/samples runs/v4_rerun runs/v4_scored runs/v3_scored \
         article/final/svg-fig-v4 runs/v4_rerun/_mechanism
# analysis scripts write their outputs under article/final/ in the authoring
# layout; created here so regenerated tables and figures land inside the package

ln -sfn "$here/data/bank/bank_full.yaml"           pexpo_bench/samples/pexpo_bench_v3_full.patched_20260811.yaml
ln -sfn "$here/data/bank/bank_evaluation_set.yaml" pexpo_bench/samples/pexpo_bench_v3_release.patched_20260811.yaml
ln -sfn "$here/data/bank/bank_calculation_stream.yaml" pexpo_bench/samples/tool_required_100q.yaml

for m in data/trajectories/main/*; do
  [ -d "$m" ] && ln -sfn "$here/$m" "runs/v4_rerun/$(basename "$m")"
done
ln -sfn "$here/data/judges/open_ended_judgments.jsonl" runs/v4_rerun/_open_judge
mkdir -p runs/v4_rerun/_hr
ln -sfn "$here/data/judges/grounding_judgments.jsonl" runs/v4_rerun/_hr/per_row_hr.jsonl

ln -sfn "$here/data/scored/results_main.parquet"        runs/v4_scored/all_scored_v4_main.parquet
ln -sfn "$here/data/scored/results_all_phases.parquet"  runs/v4_scored/all_scored_v4.parquet

ln -sfn "$here/data/trajectories/factorial" runs/v4_factorial
ln -sfn "$here/data/trajectories/seeds"     runs/v4_seeds
ln -sfn "$here/data/original_campaign/original_campaign_scored.parquet" runs/v3_scored/all_scored_v2.parquet


# the programmatic (tool-sensitive) stream is carried inside the full bank; the
# analysis scripts expect it as a separate file, so derive it here.
python3 - <<'PY'
import pathlib, yaml
root = pathlib.Path(".")
full = yaml.safe_load((root / "data/bank/bank_full.yaml").read_text())
ts = [q for q in full if str(q.get("qid", "")).startswith("TS")]
out = root / "pexpo_bench/samples/tool_required_extension_v3.yaml"
out.write_text(yaml.safe_dump(ts, allow_unicode=True, sort_keys=False))
print(f"derived {out} ({len(ts)} items)")
PY

echo "paths prepared. Now: export PEXPO_ROOT=$here"
