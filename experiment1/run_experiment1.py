"""Run registered Experiment 1 model suites with matched seeds and audit metadata.

The runner creates immutable per-run INI files under ``experiment1/generated``
and invokes the repository's ``run.py``.  It never substitutes an unavailable
ablation: unavailable variants are recorded in ``blocked_variants.json``.
"""
from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def timestamp() -> str:
    """ISO-8601 UTC timestamp used for grep-friendly, server-independent logs."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_with_timestamped_log(command: list[str], cwd: Path, log_path: Path) -> int:
    """Stream a child process into a durable log with a timestamp per line."""
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"[{timestamp()}] RUN {' '.join(command)}\n")
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            # tqdm uses carriage returns; normalize them so the saved log
            # remains one timestamped record per rendered progress update.
            for record in line.replace("\r", "\n").splitlines():
                log.write(f"[{timestamp()}] {record}\n")
        return_code = process.wait()
        log.write(f"[{timestamp()}] FINISHED return_code={return_code}\n")
    return return_code


def load_manifest() -> dict:
    return json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def registered_models() -> set[str]:
    # Keep an explicit mapping to setup() identifiers.  In particular, the
    # experiment alias ``mult`` uses the registered ``multimodal-transformer``.
    return {"qrsan", "qsan", "qrsan-no-residual", "qdnn-ablation", "real-qrsan", "real-imag-concat-mlp", "tfn", "lmf", "qmf", "mult"}


def write_config(base: Path, output: Path, dataset: str, network_type: str, seed: int, run_dir: Path, epochs: int | None, overrides: dict[str, object]) -> None:
    parser = configparser.ConfigParser()
    parser.read(base, encoding="utf-8")
    common = parser["COMMON"]
    common["dataset_name"] = dataset
    common["label"] = "emotion" if dataset == "iemocap" else "sentiment"
    common["network_type"] = "multimodal-transformer" if network_type == "mult" else network_type
    common["seed"] = str(seed)
    common["dir_name"] = str(run_dir.relative_to(ROOT))
    common["output_file"] = str((run_dir / "metrics.csv").relative_to(ROOT))
    common["prediction_file"] = str((run_dir / "predictions.npz").relative_to(ROOT))
    common["retain_model_file"] = str((run_dir / "best_model.pt").relative_to(ROOT))
    if epochs is not None:
        common["epochs"] = str(epochs)
    for key, value in overrides.items():
        common[key] = str(value).lower() if isinstance(value, bool) else str(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        parser.write(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("main", "ablation_available", "projection_head", "projection_mapping", "basis_sensitivity", "efficiency"), required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--models", nargs="+", help="Variant names to run (defaults to every variant in the suite).")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", default="eval/experiment1", help="Relative directory for suite outputs.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--list-jobs", action="store_true", help="Print valid jobs as tab-separated seed, dataset, model rows and exit.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    datasets = args.datasets or manifest["datasets"]
    seeds = args.seeds or manifest["seeds"]
    models = manifest["experiments"][args.suite]
    allowed = registered_models()
    specs = [entry if isinstance(entry, dict) else {"name": entry, "network_type": entry} for entry in models]
    if args.models:
        requested = set(args.models)
        known = {spec["name"] for spec in specs}
        unknown = sorted(requested - known)
        if unknown:
            parser.error("unknown model(s) for %s: %s" % (args.suite, ", ".join(unknown)))
        specs = [spec for spec in specs if spec["name"] in requested]
    unavailable = sorted({spec["network_type"] for spec in specs} - allowed)
    relative_output_root = Path(args.output_root)
    if relative_output_root.is_absolute() or ".." in relative_output_root.parts:
        parser.error("--output-root must be a relative path inside the repository")
    output_root = ROOT / relative_output_root / args.suite
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "blocked_variants.json").write_text(json.dumps({
        "suite": args.suite, "unavailable": unavailable,
        "required_but_not_implemented": manifest["required_but_not_implemented"],
    }, indent=2) + "\n", encoding="utf-8")
    if args.list_jobs:
        for seed in seeds:
            for dataset in datasets:
                for spec in specs:
                    if spec["network_type"] in allowed:
                        print(f"{seed}\t{dataset}\t{spec['name']}")
        return 0
    failures = []
    for seed in seeds:
        for dataset in datasets:
            for spec in specs:
                model = spec["name"]
                network_type = spec["network_type"]
                if network_type not in allowed:
                    continue
                run_dir = output_root / f"seed_{seed}" / dataset / model
                result = run_dir / "metrics.csv"
                prediction = run_dir / "predictions.npz"
                if not args.no_resume and result.exists() and prediction.exists():
                    print(f"SKIP {args.suite} seed={seed} {dataset}/{model}", flush=True)
                    continue
                config = HERE / "generated" / args.suite / f"seed_{seed}" / dataset / f"{model}.ini"
                base_name = "qrsan" if network_type in {"qsan", "qrsan-no-residual", "qdnn-ablation", "qrsan"} else network_type
                write_config(ROOT / "config" / "reproduction" / f"{base_name}.ini", config, dataset, network_type, seed, run_dir, args.epochs, spec.get("overrides", {}))
                command = [sys.executable, "-u", "run.py", "-config", str(config)]
                print("RUN " + " ".join(command), flush=True)
                if args.dry_run:
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                started_at = timestamp()
                started = time.perf_counter()
                return_code = run_with_timestamped_log(command, ROOT, run_dir / "run.log")
                (run_dir / "runtime.json").write_text(json.dumps({"started_at_utc": started_at, "finished_at_utc": timestamp(), "wall_seconds": time.perf_counter() - started, "return_code": return_code}, indent=2) + "\n", encoding="utf-8")
                if return_code:
                    failures.append(f"seed_{seed}/{dataset}/{model}")
    if failures:
        print("FAILED: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
