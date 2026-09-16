"""Run the main, factorial or seed-replication design reported in the paper."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex

from pexpo_bench.data_schema import MAIN_CONFIGURATIONS, SUBJECT_MODELS


def experiment_commands(experiment, root, output, models, concurrency=10):
    """Build explicit commands; the main run supplies the factorial end points."""
    designs = {
        "main": [("bank_evaluation_set.yaml", MAIN_CONFIGURATIONS, 42, "main")],
        "factorial": [("bank_calculation_stream.yaml",
                       ("F100", "F010", "F001", "F110", "F101", "F011"),
                       42, "factorial")],
        "seeds": [("bank_evaluation_set.yaml", ("A3", "A4"), seed, f"seeds/seed{seed}")
                  for seed in (43, 44, 45)],
    }
    commands = []
    for bank, configurations, seed, directory in designs[experiment]:
        commands.append([
            "--bank", str(root / "data/bank" / bank),
            "--out", str(output / directory),
            "--models", *models, "--archs", *configurations,
            "--seed", str(seed), "--temperature", "0.3", "--run-idx", "1",
            "--concurrency", str(concurrency),
        ])
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("main", "factorial", "seeds"), required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", choices=SUBJECT_MODELS, default=list(SUBJECT_MODELS))
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without API calls")
    args = parser.parse_args(argv)
    root = Path(os.environ.get("PEXPO_ROOT", Path(__file__).resolve().parents[3]))
    for command in experiment_commands(args.experiment, root, args.out, args.models, args.concurrency):
        print("python -m pexpo_bench.runners.run_experiment " + shlex.join(command), flush=True)
        if not args.dry_run:
            from pexpo_bench.runners.run_experiment import main as run
            code = run(command)
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
