"""Train five full-QRSAN runs per dataset and save Figure 4 predictions."""

from __future__ import annotations

import argparse
import sys

from train_figures import DATASETS, Experiment, check_experiment, make_config, run_experiment


def build_experiments(runs: int, base_seed: int) -> list[Experiment]:
    experiments = []
    for run_id in range(1, runs + 1):
        for dataset in DATASETS:
            experiments.append(Experiment(
                "figure4_predictions", dataset, "T+V+A", "textual,visual,acoustic",
                "Full QRSAN", run_id, base_seed + run_id - 1, "qrsan", True,
            ))
    return experiments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.worker_count < 1:
        parser.error("--runs and --worker-count must be positive")
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("--worker-index must be in [0, --worker-count)")

    experiments = build_experiments(args.runs, args.base_seed)
    assigned = experiments[args.worker_index::args.worker_count]
    print(f"worker {args.worker_index}/{args.worker_count}: {len(assigned)}/{len(experiments)} experiments", flush=True)
    failures = 0
    for experiment in assigned:
        config_path, result_path, _ = make_config(experiment, args.epochs)
        prediction_path = result_path.parent / "predictions.npz"
        if args.dry_run:
            print(f"DRY-RUN {config_path} -> {result_path}", flush=True)
            continue
        code = (check_experiment(config_path) if args.check
                # A metrics CSV alone cannot be used for Figure 4.  Retrain if
                # raw test predictions were not saved with that run.
                else run_experiment(
                    experiment, args.epochs,
                    not args.no_resume and prediction_path.exists(),
                ))
        if code:
            failures += 1
            print(f"FAILED return_code={code}: {config_path}", flush=True)
            if not args.continue_on_error:
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
