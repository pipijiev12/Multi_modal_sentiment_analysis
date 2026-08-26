"""Read-only integrity audit for completed Experiment 1 result artifacts.

By default, only jobs with both metrics.csv and predictions.npz are audited;
pending/running jobs are reported but do not fail the command.  Add
``--require-all`` only after the batch has finished to require all registered
jobs to be complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def audit_prediction(path: Path) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    detail: dict[str, object] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = {"outputs", "targets"} - set(archive.files)
            if missing:
                return ["missing_npz_keys:" + ",".join(sorted(missing))], detail
            outputs, targets = np.asarray(archive["outputs"]), np.asarray(archive["targets"])
    except Exception as error:  # corrupt archive must be reported, not crash the audit
        return [f"unreadable_npz:{type(error).__name__}"], detail
    detail.update({
        "n_test_instances": int(targets.shape[0]) if targets.ndim else 0,
        "output_shape": list(outputs.shape), "target_shape": list(targets.shape),
        "target_sha256": sha256(targets),
    })
    if outputs.ndim == 0 or targets.ndim == 0 or outputs.shape[0] != targets.shape[0] or targets.shape[0] == 0:
        errors.append("invalid_prediction_target_leading_dimension")
    if not np.issubdtype(outputs.dtype, np.number) or not np.issubdtype(targets.dtype, np.number):
        errors.append("non_numeric_prediction_or_target")
    elif not np.isfinite(outputs).all() or not np.isfinite(targets).all():
        errors.append("non_finite_prediction_or_target")
    return errors, detail


def audit_metrics(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
    except Exception as error:
        return [f"unreadable_metrics_csv:{type(error).__name__}"]
    if len(rows) < 2 or not rows[0] or not any(cell.strip() for cell in rows[0]):
        return ["empty_or_header_only_metrics_csv"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="eval/experiment1/audits/result_integrity")
    parser.add_argument("--suite", action="append", help="Limit audit to a suite; repeatable.")
    parser.add_argument("--require-all", action="store_true", help="Fail if any registered job lacks complete artifacts.")
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    selected = set(args.suite or manifest["experiments"])
    unknown = selected - set(manifest["experiments"])
    if unknown:
        parser.error("unknown suite(s): " + ", ".join(sorted(unknown)))

    rows: list[dict[str, object]] = []
    target_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    summary: Counter[str] = Counter()
    for suite, entries in manifest["experiments"].items():
        if suite not in selected or suite == "basis_sensitivity":
            continue
        specs = [entry if isinstance(entry, dict) else {"name": entry} for entry in entries]
        for seed in manifest["seeds"]:
            for dataset in manifest["datasets"]:
                for spec in specs:
                    model = spec["name"]
                    run_dir = ROOT / "eval" / "experiment1" / suite / f"seed_{seed}" / dataset / model
                    metrics, prediction = run_dir / "metrics.csv", run_dir / "predictions.npz"
                    row: dict[str, object] = {"suite": suite, "seed": seed, "dataset": dataset, "model": model, "run_dir": str(run_dir)}
                    if not metrics.is_file() or not prediction.is_file():
                        row.update({"status": "missing_artifacts", "errors": "metrics.csv_or_predictions.npz_missing"})
                        summary["missing_artifacts"] += 1
                        rows.append(row)
                        continue
                    errors = audit_metrics(metrics)
                    prediction_errors, detail = audit_prediction(prediction)
                    errors.extend(prediction_errors)
                    runtime = run_dir / "runtime.json"
                    if runtime.is_file():
                        try:
                            return_code = json.loads(runtime.read_text(encoding="utf-8")).get("return_code")
                            if return_code != 0:
                                errors.append(f"nonzero_return_code:{return_code}")
                        except Exception:
                            errors.append("unreadable_runtime_json")
                    else:
                        row["warning"] = "runtime.json_missing"
                    row.update(detail)
                    row["status"] = "valid" if not errors else "invalid"
                    row["errors"] = ";".join(errors)
                    summary[row["status"]] += 1
                    if "target_sha256" in detail:
                        target_groups[(suite, dataset)].add(str(detail["target_sha256"]))
                    rows.append(row)

    for row in rows:
        key = (str(row["suite"]), str(row["dataset"]))
        if row["status"] == "valid" and len(target_groups[key]) > 1:
            row["status"] = "invalid"
            row["errors"] = ";".join(filter(None, [str(row.get("errors", "")), "inconsistent_targets_within_suite_dataset"]))
            summary["valid"] -= 1
            summary["invalid"] += 1

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output / "result_integrity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    invalid = sum(1 for row in rows if row["status"] == "invalid")
    missing = sum(1 for row in rows if row["status"] == "missing_artifacts")
    report = {
        "registered_jobs": len(rows), "valid_completed_jobs": int(summary["valid"]),
        "invalid_completed_jobs": invalid, "missing_artifact_jobs": missing,
        "require_all": args.require_all, "target_hash_scope": "suite + dataset",
    }
    (output / "result_integrity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "result_integrity.md").write_text(
        "# Experiment 1 result-integrity audit\n\n"
        f"- Registered jobs: {len(rows)}\n- Valid completed jobs: {summary['valid']}\n"
        f"- Invalid completed jobs: {invalid}\n- Jobs without both required artifacts: {missing}\n"
        "\nA completed job is valid only if its metrics CSV is non-empty and its prediction archive has numeric, finite `outputs` and `targets` with equal non-zero leading dimensions.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 1 if invalid or (args.require_all and missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
