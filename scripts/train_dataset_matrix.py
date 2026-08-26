"""Train the 12-model reproduction matrix across MOSI, MOSEI and IEMOCAP."""

from __future__ import annotations

import argparse
import configparser
import json
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_DIR = ROOT / "config" / "reproduction"

DATASETS = ("cmumosi", "cmumosei", "iemocap")
MODELS = (
    "ef_lstm", "lf_lstm", "marn", "mfn", "tfn", "lmf",
    "mult", "qmf", "qrsan", "megakan", "m3sa", "almt",
)


def select(values: list[str] | None, allowed: tuple[str, ...], kind: str) -> list[str]:
    if not values or values == ["all"]:
        return list(allowed)
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown {kind}(s): {', '.join(unknown)}")
    return values


def batch_suffix(batch_id: str | None) -> str:
    if batch_id is None:
        return "matrix"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", batch_id):
        raise SystemExit(
            "Invalid --batch-id. Use only letters, numbers, '.', '_' or '-'."
        )
    return f"matrix_{batch_id}"


def make_config(
    dataset: str, model: str, config_dir: Path, run_name: str,
    epochs: int | None = None, seed: int | None = None,
) -> Path:
    parser = configparser.ConfigParser()
    parser.read(BASE_CONFIG_DIR / f"{model}.ini", encoding="utf-8")
    common = parser["COMMON"]
    common["dataset_name"] = dataset
    common["label"] = "emotion" if dataset == "iemocap" else "sentiment"
    if epochs is not None:
        common["epochs"] = str(epochs)
    if seed is not None:
        common["seed"] = str(seed)
    common["dir_name"] = f"{run_name}/{dataset}/{model}"
    common["output_file"] = f"eval/{run_name}/{dataset}/{model}.csv"
    # Keep test-set raw outputs and targets so class-level metrics (Figure 4)
    # can be reproduced exactly after all repeated runs have finished.
    common["prediction_file"] = f"eval/{run_name}/{dataset}/{model}.predictions.npz"
    target_dir = config_dir / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{model}.ini"
    with target.open("w", encoding="utf-8") as stream:
        parser.write(stream)
    return target


def prepare_configs(
    datasets: list[str], models: list[str], config_dir: Path,
    result_dir: Path, run_name: str, epochs: int | None, seed: int | None,
) -> None:
    for dataset in datasets:
        for model in models:
            make_config(
                dataset, model, config_dir, run_name, epochs, seed
            )


def check_matrix(
    datasets: list[str], models: list[str], config_dir: Path,
    result_dir: Path, run_name: str, epochs: int | None, seed: int | None,
) -> int:
    sys.path.insert(0, str(ROOT))
    from dataset import setup as setup_dataset
    from models import setup as setup_model
    from utils.model import get_criterion, get_loss
    from utils.params import Params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = 0
    print(f"Device: {device}", flush=True)
    for dataset in datasets:
        for model_name in models:
            try:
                params = Params()
                params.parse_config(str(make_config(
                    dataset, model_name, config_dir, run_name, epochs, seed,
                )))
                params.device = device
                reader = setup_dataset(params)
                reader.read(params)
                params.reader = reader
                model = setup_model(params).to(device)
                batch = next(iter(reader.get_data(iterable=True, shuffle=False, split="train")))
                inputs = [tensor.to(device) for tensor in batch[:-1]]
                target = batch[-1].to(device)
                output = model(inputs)
                if output.shape != target.shape:
                    output = output.reshape_as(target)
                loss = get_loss(params, get_criterion(params), output, target)
                if hasattr(model, "auxiliary_loss"):
                    loss = loss + model.auxiliary_loss(target)
                loss.backward()
                print(
                    f"PASS {dataset:9s}/{model_name:8s} "
                    f"target={tuple(target.shape)} loss={loss.item():.4f}",
                    flush=True,
                )
                del model, inputs, target, output, loss, reader, params
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as error:
                failures += 1
                print(f"FAIL {dataset}/{model_name}: {type(error).__name__}: {error}", flush=True)
                traceback.print_exc()
    return failures


def load_state(state_path: Path) -> dict[str, object]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"created_at": datetime.now().isoformat(timespec="seconds"), "runs": {}}


def save_state(state: dict[str, object], result_dir: Path, state_path: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train_matrix(
    datasets: list[str], models: list[str], epochs: int | None,
    continue_on_error: bool, resume: bool, config_dir: Path,
    result_dir: Path, state_path: Path, run_name: str, seed: int | None,
) -> int:
    state = load_state(state_path)
    state["last_started_at"] = datetime.now().isoformat(timespec="seconds")
    state["epochs_override"] = epochs
    failures = 0
    total = len(datasets) * len(models)
    run_index = 0
    for dataset in datasets:
        dataset_dir = result_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for model_name in models:
            run_index += 1
            key = f"{dataset}/{model_name}"
            result_path = dataset_dir / f"{model_name}.csv"
            prediction_path = dataset_dir / f"{model_name}.predictions.npz"
            if resume and result_path.exists() and prediction_path.exists():
                print(f"[{run_index}/{total}] SKIP completed {key}", flush=True)
                state["runs"][key] = {"status": "completed", "result": str(result_path)}
                save_state(state, result_dir, state_path)
                continue
            if resume and result_path.exists() and not prediction_path.exists():
                print(
                    f"[{run_index}/{total}] RERUN {key}: result CSV exists but "
                    "the required prediction artifact is missing",
                    flush=True,
                )
            config_path = make_config(
                dataset, model_name, config_dir, run_name, epochs, seed,
            )
            log_path = dataset_dir / f"{model_name}.log"
            print(f"[{run_index}/{total}] TRAIN {key}", flush=True)
            state["runs"][key] = {
                "status": "running", "config": str(config_path), "log": str(log_path)
            }
            save_state(state, result_dir, state_path)
            command = [sys.executable, "-u", "run.py", "-config", str(config_path)]
            with log_path.open("w", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_stream.write(line)
                    log_stream.flush()
                return_code = process.wait()
            status = "completed" if return_code == 0 else "failed"
            state["runs"][key] = {
                "status": status, "return_code": return_code,
                "config": str(config_path), "log": str(log_path),
                "result": str(result_path) if result_path.exists() else None,
            }
            save_state(state, result_dir, state_path)
            if return_code:
                failures += 1
                if not continue_on_error:
                    state["last_finished_at"] = datetime.now().isoformat(timespec="seconds")
                    save_state(state, result_dir, state_path)
                    return failures
    state["last_finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state, result_dir, state_path)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    datasets = select(args.datasets, DATASETS, "dataset")
    models = select(args.models, MODELS, "model")
    run_name = batch_suffix(args.batch_id)
    config_dir = ROOT / "config" / run_name
    result_dir = ROOT / "eval" / run_name
    state_path = result_dir / "training_state.json"
    prepare_configs(
        datasets, models, config_dir, result_dir,
        run_name, args.epochs, args.seed,
    )
    if args.check:
        return 1 if check_matrix(
            datasets, models, config_dir, result_dir,
            run_name, args.epochs, args.seed,
        ) else 0
    return 1 if train_matrix(
        datasets, models, args.epochs, args.continue_on_error,
        not args.no_resume, config_dir, result_dir, state_path,
        run_name, args.seed,
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
