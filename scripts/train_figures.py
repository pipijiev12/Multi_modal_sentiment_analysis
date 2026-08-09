"""Run the missing repeated experiments for Figures 5 and 6.

Figure 5: six missing QRSAN modality settings (T, V, A, T+V, T+A, V+A).
Figure 6: three-modal ablation of Basic QDNN, QSAN self-attention, and
residual QRSAN.
"""

from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config" / "reproduction" / "qrsan.ini"
DATASETS = ("cmumosei", "cmumosi", "iemocap")
FIGURE5_MODALITIES = (
    ("T", "textual"),
    ("V", "visual"),
    ("A", "acoustic"),
    ("T+V", "textual,visual"),
    ("T+A", "textual,acoustic"),
    ("V+A", "visual,acoustic"),
)


@dataclass(frozen=True)
class Experiment:
    figure: str
    dataset: str
    modality_name: str
    features: str
    variant: str
    run_id: int
    seed: int
    network_type: str
    residual: bool | None

    @property
    def slug(self) -> str:
        return self.modality_name.replace("+", "_").lower()


def selected_figures(value: str) -> tuple[str, ...]:
    return ("figure5", "figure6") if value == "all" else (value,)


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    for run_id in range(1, args.runs + 1):
        seed = args.base_seed + run_id - 1
        for figure in selected_figures(args.figure):
            if figure == "figure5":
                for dataset in DATASETS:
                    for modality_name, features in FIGURE5_MODALITIES:
                        experiments.append(Experiment(
                            figure, dataset, modality_name, features,
                            "Full QRSAN", run_id, seed, "qrsan", True,
                        ))
            else:
                variants = [
                    ("Basic QDNN", "qdnn", None),
                    ("QSAN", "qsan", False),
                    ("Full QRSAN", "qrsan", True),
                ]
                for dataset in DATASETS:
                    for variant, network_type, residual in variants:
                        experiments.append(Experiment(
                            figure, dataset, "T+V+A", "textual,visual,acoustic",
                            variant, run_id, seed, network_type, residual,
                        ))
    return experiments


def make_config(experiment: Experiment, epochs: int | None) -> tuple[Path, Path, Path]:
    parser = configparser.ConfigParser()
    parser.read(TEMPLATE, encoding="utf-8")
    common = parser["COMMON"]
    run_dir = (
        ROOT / "eval" / experiment.figure / experiment.dataset /
        experiment.slug / experiment.variant.replace(" ", "_").lower() /
        f"run_{experiment.run_id}"
    )
    config_path = (
        ROOT / "config" / experiment.figure / experiment.dataset /
        experiment.slug / experiment.variant.replace(" ", "_").lower() /
        f"run_{experiment.run_id}.ini"
    )
    result_path = run_dir / "metrics.csv"
    log_path = run_dir / "train.log"
    common["dataset_name"] = experiment.dataset
    common["label"] = "emotion" if experiment.dataset == "iemocap" else "sentiment"
    common["features"] = experiment.features
    common["network_type"] = experiment.network_type
    if experiment.residual is not None:
        common["residual_self_attention"] = str(experiment.residual).lower()
    common["seed"] = str(experiment.seed)
    common["run_id"] = str(experiment.run_id)
    common["variant"] = experiment.variant
    common["experiment_figure"] = experiment.figure
    if epochs is not None:
        common["epochs"] = str(epochs)
    common["dir_name"] = str(run_dir.relative_to(ROOT / "eval"))
    common["output_file"] = str(result_path.relative_to(ROOT))
    common["prediction_file"] = str((run_dir / "predictions.npz").relative_to(ROOT))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as stream:
        parser.write(stream)
    return config_path, result_path, log_path


def run_experiment(experiment: Experiment, epochs: int | None, resume: bool) -> int:
    config_path, result_path, log_path = make_config(experiment, epochs)
    label = (
        f"{experiment.figure} {experiment.dataset} {experiment.modality_name} "
        f"{experiment.variant} run={experiment.run_id} seed={experiment.seed}"
    )
    if resume and result_path.exists():
        print(f"SKIP {label}", flush=True)
        return 0
    result_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"TRAIN {label}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            [sys.executable, "-u", "run.py", "-config", str(config_path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
        return process.wait()


def check_experiment(config_path: Path) -> int:
    """Validate one forward/backward batch without writing a result CSV."""
    import torch

    sys.path.insert(0, str(ROOT))
    from dataset import setup as setup_dataset
    from models import setup as setup_model
    from utils.model import get_criterion, get_loss
    from utils.params import Params

    params = Params()
    params.parse_config(str(config_path))
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = setup_dataset(params)
    reader.read(params)
    params.reader = reader
    model = setup_model(params).to(params.device)
    batch = next(iter(reader.get_data(iterable=True, shuffle=False, split="train")))
    inputs = [value.to(params.device) for value in batch[:-1]]
    target = batch[-1].to(params.device)
    output = model(inputs)
    if output.shape != target.shape:
        output = output.reshape_as(target)
    loss = get_loss(params, get_criterion(params), output, target)
    loss.backward()
    print(f"PASS {config_path}: loss={loss.item():.4f}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("figure5", "figure6", "all"), default="all")
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

    experiments = build_experiments(args)
    assigned = experiments[args.worker_index::args.worker_count]
    print(
        f"worker {args.worker_index}/{args.worker_count}: "
        f"{len(assigned)}/{len(experiments)} experiments",
        flush=True,
    )
    failures = 0
    for experiment in assigned:
        config_path, result_path, _ = make_config(experiment, args.epochs)
        if args.dry_run:
            print(f"DRY-RUN {config_path} -> {result_path}", flush=True)
            continue
        return_code = (
            check_experiment(config_path) if args.check
            else run_experiment(experiment, args.epochs, not args.no_resume)
        )
        if return_code:
            failures += 1
            print(f"FAILED return_code={return_code}: {config_path}", flush=True)
            if not args.continue_on_error:
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
