"""Check or train the paper-model reproduction suite on CMU-MOSI."""

from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config" / "reproduction"
RESULT_DIR = ROOT / "eval" / "reproduction"
TMP_CONFIG_DIR = ROOT / "tmp" / "reproduction_configs"
MODEL_ORDER = (
    "ef_lstm",
    "lf_lstm",
    "marn",
    "mfn",
    "tfn",
    "lmf",
    "mult",
    "qmf",
    "qrsan",
    "megakan",
    "m3sa",
    "almt",
)


def select_models(names: list[str] | None) -> list[str]:
    if not names or names == ["all"]:
        return list(MODEL_ORDER)
    unknown = sorted(set(names) - set(MODEL_ORDER))
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")
    return names


def materialize_config(model_name: str, epochs: int | None) -> Path:
    source = CONFIG_DIR / f"{model_name}.ini"
    if epochs is None:
        return source
    parser = configparser.ConfigParser()
    parser.read(source, encoding="utf-8")
    parser["COMMON"]["epochs"] = str(epochs)
    parser["COMMON"]["output_file"] = (
        f"eval/reproduction/{epochs}ep/cmumosi_{model_name}.csv"
    )
    parser["COMMON"]["dir_name"] = f"reproduction/{epochs}ep/{model_name}_cmumosi"
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    target = TMP_CONFIG_DIR / f"{model_name}_{epochs}ep.ini"
    with target.open("w", encoding="utf-8") as stream:
        parser.write(stream)
    return target


def check_models(model_names: list[str]) -> int:
    sys.path.insert(0, str(ROOT))
    from dataset import setup as setup_dataset
    from models import setup as setup_model
    from utils.params import Params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = 0
    print(f"Device: {device}", flush=True)
    for model_name in model_names:
        try:
            params = Params()
            params.parse_config(str(CONFIG_DIR / f"{model_name}.ini"))
            params.device = device
            reader = setup_dataset(params)
            reader.read(params)
            params.reader = reader
            model = setup_model(params).to(device)
            batch = next(iter(reader.get_data(iterable=True, shuffle=False, split="train")))
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            target = batch[-1].to(device)
            output = model(inputs).reshape_as(target)
            loss = torch.nn.functional.l1_loss(output, target)
            loss.backward()
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            print(
                f"PASS {model_name:8s} output={tuple(output.shape)} "
                f"params={parameter_count:,} loss={loss.item():.4f}",
                flush=True,
            )
            del model, inputs, target, output, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:  # report every model in one check run
            failures += 1
            print(f"FAIL {model_name:8s} {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
    return failures


def train_models(model_names: list[str], epochs: int | None, continue_on_error: bool) -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(exist_ok=True)
    state_path = RESULT_DIR / "training_state.json"
    state: dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "epochs_override": epochs,
        "models": {},
    }
    failures = 0
    for index, model_name in enumerate(model_names, start=1):
        config_path = materialize_config(model_name, epochs)
        log_path = RESULT_DIR / f"{model_name}.log"
        print(f"[{index}/{len(model_names)}] Training {model_name}", flush=True)
        state["models"][model_name] = {"status": "running", "log": str(log_path)}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [sys.executable, "-u", "run.py", "-config", str(config_path)]
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command,
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
            return_code = process.wait()
        status = "completed" if return_code == 0 else "failed"
        state["models"][model_name] = {
            "status": status,
            "return_code": return_code,
            "log": str(log_path),
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        if return_code:
            failures += 1
            if not continue_on_error:
                break
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names or 'all'.")
    parser.add_argument("--check", action="store_true", help="Run one forward/backward batch per model.")
    parser.add_argument("--epochs", type=int, help="Temporarily override epochs in every config.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    models = select_models(args.models)
    if args.check:
        return 1 if check_models(models) else 0
    return 1 if train_models(models, args.epochs, args.continue_on_error) else 0


if __name__ == "__main__":
    raise SystemExit(main())
