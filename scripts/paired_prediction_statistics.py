"""Paired bootstrap and permutation tests for QRSAN prediction files.

For every pre-specified dataset/comparator pair, the script loads five matched
seed runs from ``<input-root>/matrix_<id>/<dataset>/<model>.predictions.npz``.
It estimates the mean QRSAN-minus-baseline difference by paired bootstrap,
tests it with a two-sided paired permutation test, and Holm-corrects p-values
across every requested dataset/metric comparison in one analysis invocation.

The comparator mapping is deliberately supplied by the caller rather than
selected from the same test-set scores.  This keeps the target comparison
explicit and avoids silently introducing post-hoc selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DATASETS = ("cmumosei", "cmumosi", "iemocap")
SENTIMENT_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
IEMOCAP_METRICS = SENTIMENT_METRICS
DEFAULT_COMPARATORS = {
    "cmumosei": "almt",
    "cmumosi": "megakan",
    "iemocap": "ef_lstm",
}
DISPLAY_DATASET = {
    "cmumosei": "CMU-MOSEI",
    "cmumosi": "CMU-MOSI",
    "iemocap": "IEMOCAP",
}
DISPLAY_MODEL = {
    "qrsan": "QRSAN",
    "almt": "ALMT",
    "megakan": "MEGAKANs",
    "ef_lstm": "EF-LSTM",
}
DISPLAY_METRIC = {
    "accuracy": "ACC",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro-F1",
}


@dataclass(frozen=True)
class PredictionRun:
    """Discrete predictions and labels, grouped by original test instance."""

    prediction: np.ndarray
    target: np.ndarray
    source: Path

    @property
    def n_instances(self) -> int:
        return int(self.target.shape[0])


@dataclass(frozen=True)
class PairedRun:
    run_id: str
    qrsan: PredictionRun
    baseline: PredictionRun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="eval")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-ids", nargs="+", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--comparators", nargs="+",
        default=[f"{dataset}={model}" for dataset, model in DEFAULT_COMPARATORS.items()],
        metavar="DATASET=MODEL",
        help="Pre-specified comparator mapping, for example cmumosei=almt.",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=SENTIMENT_METRICS,
        default=list(SENTIMENT_METRICS),
        help="For IEMOCAP, balanced accuracy and macro-F1 are macro-averaged over four emotion labels.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--permutation-resamples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260818)
    parser.add_argument("--table-label", default="Sx")
    args = parser.parse_args()
    if args.bootstrap_resamples < 10000:
        parser.error("--bootstrap-resamples must be at least 10,000")
    if args.permutation_resamples < 10000:
        parser.error("--permutation-resamples must be at least 10,000")
    if len(set(args.batch_ids)) != len(args.batch_ids):
        parser.error("--batch-ids must be unique")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.table_label):
        parser.error("--table-label must contain only letters, numbers, '.', '_' or '-'")
    return args


def parse_comparators(values: list[str], datasets: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid comparator {value!r}; expected DATASET=MODEL")
        dataset, model = value.split("=", 1)
        if dataset not in DATASETS or not model or not re.fullmatch(r"[A-Za-z0-9_.-]+", model):
            raise ValueError(f"invalid comparator {value!r}")
        if dataset in mapping:
            raise ValueError(f"duplicate comparator for {dataset}")
        mapping[dataset] = model
    missing = sorted(set(datasets) - set(mapping))
    if missing:
        raise ValueError("missing comparator mapping for: " + ", ".join(missing))
    return mapping


def run_id(batch_id: str) -> str:
    return f"matrix_{batch_id}"


def prediction_path(input_root: Path, batch_id: str, dataset: str, model: str) -> Path:
    return input_root / run_id(batch_id) / dataset / f"{model}.predictions.npz"


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        missing = {"outputs", "targets"} - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing array(s): {', '.join(sorted(missing))}")
        return np.asarray(archive["outputs"]), np.asarray(archive["targets"])


def discrete_sentiment(outputs: np.ndarray, targets: np.ndarray, source: Path) -> PredictionRun:
    prediction = np.asarray(outputs).reshape(-1) >= 0
    target = np.asarray(targets).reshape(-1) >= 0
    if prediction.shape != target.shape:
        raise ValueError(f"{source}: sentiment prediction/target shapes differ")
    return PredictionRun(prediction=prediction, target=target, source=source)


def discrete_iemocap(outputs: np.ndarray, targets: np.ndarray, source: Path) -> PredictionRun:
    # ``utils.model.test`` stores model output before reshaping.  Some models
    # therefore save (N, 8), while targets are stored as (N, 4, 2).
    if outputs.ndim == 2 and outputs.shape[1] == 8:
        outputs = outputs.reshape(-1, 4, 2)
    if targets.ndim == 2 and targets.shape[1] == 8:
        targets = targets.reshape(-1, 4, 2)
    if outputs.ndim != 3 or targets.ndim != 3 or outputs.shape != targets.shape:
        raise ValueError(
            f"{source}: expected matching IEMOCAP arrays shaped (N, 4, 2), got "
            f"outputs={outputs.shape}, targets={targets.shape}"
        )
    if outputs.shape[1:] != (4, 2):
        raise ValueError(f"{source}: expected IEMOCAP trailing shape (4, 2)")
    return PredictionRun(
        prediction=outputs.argmax(axis=-1),
        target=targets.argmax(axis=-1),
        source=source,
    )


def load_run(path: Path, dataset: str) -> PredictionRun:
    outputs, targets = load_archive(path)
    if dataset == "iemocap":
        return discrete_iemocap(outputs, targets, path)
    return discrete_sentiment(outputs, targets, path)


def metric_value(metric: str, prediction: np.ndarray, target: np.ndarray, dataset: str) -> float:
    """Evaluate a metric on original instances or a paired bootstrap sample."""
    if dataset == "iemocap":
        prediction = np.asarray(prediction, dtype=bool)
        target = np.asarray(target, dtype=bool)
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 4:
            raise ValueError("invalid IEMOCAP prediction/target arrays")
        if metric == "accuracy":
            return float(np.mean(prediction == target))
        per_label = []
        for label_index in range(prediction.shape[1]):
            label_prediction = prediction[:, label_index]
            label_target = target[:, label_index]
            true_negative = int(np.count_nonzero(~label_target & ~label_prediction))
            false_positive = int(np.count_nonzero(~label_target & label_prediction))
            false_negative = int(np.count_nonzero(label_target & ~label_prediction))
            true_positive = int(np.count_nonzero(label_target & label_prediction))
            negative_recall = true_negative / (true_negative + false_positive) \
                if true_negative + false_positive else 0.0
            positive_recall = true_positive / (true_positive + false_negative) \
                if true_positive + false_negative else 0.0
            negative_f1 = (2 * true_negative) / (2 * true_negative + false_positive + false_negative) \
                if 2 * true_negative + false_positive + false_negative else 0.0
            positive_f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative) \
                if 2 * true_positive + false_positive + false_negative else 0.0
            per_label.append({
                "balanced_accuracy": (negative_recall + positive_recall) / 2,
                "macro_f1": (negative_f1 + positive_f1) / 2,
            })
        return float(np.mean([row[metric] for row in per_label]))

    prediction = np.asarray(prediction, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    if prediction.shape != target.shape or target.size == 0:
        raise ValueError("invalid sentiment prediction/target arrays")
    true_negative = int(np.count_nonzero(~target & ~prediction))
    false_positive = int(np.count_nonzero(~target & prediction))
    false_negative = int(np.count_nonzero(target & ~prediction))
    true_positive = int(np.count_nonzero(target & prediction))
    if metric == "accuracy":
        return float((true_negative + true_positive) / target.size)
    negative_recall = true_negative / (true_negative + false_positive) \
        if true_negative + false_positive else 0.0
    positive_recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    if metric == "balanced_accuracy":
        return float((negative_recall + positive_recall) / 2)
    if metric == "macro_f1":
        negative_f1 = (2 * true_negative) / (
            2 * true_negative + false_positive + false_negative
        ) if 2 * true_negative + false_positive + false_negative else 0.0
        positive_f1 = (2 * true_positive) / (
            2 * true_positive + false_positive + false_negative
        ) if 2 * true_positive + false_positive + false_negative else 0.0
        return float((negative_f1 + positive_f1) / 2)
    raise ValueError(f"unsupported metric {metric}")


def verify_paired_runs(runs: list[PairedRun]) -> None:
    if not runs:
        raise ValueError("at least one matched run is required")
    reference_target = runs[0].qrsan.target
    for run in runs:
        if run.qrsan.target.shape != run.baseline.target.shape or not np.array_equal(
            run.qrsan.target, run.baseline.target,
        ):
            raise ValueError(
                f"{run.run_id}: QRSAN and baseline targets are not identical and ordered"
            )
        if run.qrsan.target.shape != reference_target.shape or not np.array_equal(
            run.qrsan.target, reference_target,
        ):
            raise ValueError(
                "all matched runs must use identical, ordered test targets"
            )


def paired_bootstrap(
    runs: list[PairedRun], dataset: str, metric: str, resamples: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Bootstrap test instances jointly across all seed-matched runs."""
    n_instances = runs[0].qrsan.n_instances
    differences = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_indices = generator.integers(0, n_instances, size=n_instances)
        seed_differences = []
        for run in runs:
            target = run.qrsan.target[sampled_indices]
            qrsan_score = metric_value(
                metric, run.qrsan.prediction[sampled_indices], target, dataset,
            )
            baseline_score = metric_value(
                metric, run.baseline.prediction[sampled_indices], target, dataset,
            )
            seed_differences.append(qrsan_score - baseline_score)
        differences[index] = float(np.mean(seed_differences))
    return differences


def paired_permutation(
    runs: list[PairedRun], dataset: str, metric: str, resamples: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Generate a paired prediction-swap null distribution for one metric."""
    n_instances = runs[0].qrsan.n_instances
    differences = np.empty(resamples, dtype=float)
    for index in range(resamples):
        seed_differences = []
        for run in runs:
            swap = generator.integers(0, 2, size=n_instances, dtype=np.int8).astype(bool)
            if dataset == "iemocap":
                swap = swap[:, np.newaxis]
            qrsan_prediction = np.where(
                swap, run.baseline.prediction, run.qrsan.prediction,
            )
            baseline_prediction = np.where(
                swap, run.qrsan.prediction, run.baseline.prediction,
            )
            qrsan_score = metric_value(
                metric, qrsan_prediction, run.qrsan.target, dataset,
            )
            baseline_score = metric_value(
                metric, baseline_prediction, run.qrsan.target, dataset,
            )
            seed_differences.append(qrsan_score - baseline_score)
        differences[index] = float(np.mean(seed_differences))
    return differences


def mean_and_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def cohen_dz(differences: list[float]) -> float | None:
    values = np.asarray(differences, dtype=float)
    if values.size < 2:
        return None
    standard_deviation = float(values.std(ddof=1))
    if math.isclose(standard_deviation, 0.0, abs_tol=1e-15):
        return 0.0 if math.isclose(float(values.mean()), 0.0, abs_tol=1e-15) else None
    return float(values.mean() / standard_deviation)


def holm_adjust(rows: list[dict[str, object]]) -> None:
    """In-place Holm correction for every metric/dataset hypothesis."""
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1]["p_value"]))
    family_size = len(ordered)
    running_maximum = 0.0
    for position, (original_index, row) in enumerate(ordered):
        adjusted = min(1.0, float(row["p_value"]) * (family_size - position))
        running_maximum = max(running_maximum, adjusted)
        rows[original_index]["adjusted_p_value"] = running_maximum


def interpretation(row: dict[str, object]) -> str:
    delta = float(row["delta"])
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    adjusted_p = float(row["adjusted_p_value"])
    if delta > 0 and lower > 0 and adjusted_p < 0.05:
        return "Superior"
    if delta < 0 and upper < 0 and adjusted_p < 0.05:
        return "Inferior"
    return "Competitive"


def format_mean_std(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def format_pp(value: float) -> str:
    return f"{100 * value:.2f}"


def format_effect(effect_size: float | None) -> str:
    return "NA" if effect_size is None else f"{effect_size:.3f}"


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, object]], table_label: str) -> None:
    lines = [
        f"# Table {table_label}. Paired statistical comparison with pre-specified baselines",
        "",
        "Differences are QRSAN minus the comparator in percentage points. Confidence intervals "
        "are percentile intervals from paired bootstrap resampling of identical test instances. "
        "Two-sided p-values are from paired prediction-swap permutation tests and are Holm-adjusted "
        "over all rows in this table. Effect size is paired Cohen's $d_z$ over the matched seed-level "
        "metric differences. `NA` indicates zero seed-pair variance with a non-zero mean difference.",
        "",
        "| Dataset | Metric | Comparator | QRSAN | Baseline | Δ (pp) | 95% CI (pp) | Adjusted p | Effect size ($d_z$) | Interpretation |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {metric} | {comparator} | {qrsan} | {baseline} | {delta} | "
            "{lower} to {upper} | {adjusted_p:.4g} | {effect} | {interpretation} |".format(
                dataset=row["dataset_display"],
                metric=row["metric_display"],
                comparator=row["comparator_display"],
                qrsan=format_mean_std(float(row["qrsan_mean"]), float(row["qrsan_std"])),
                baseline=format_mean_std(float(row["baseline_mean"]), float(row["baseline_std"])),
                delta=format_pp(float(row["delta"])),
                lower=format_pp(float(row["ci_lower"])),
                upper=format_pp(float(row["ci_upper"])),
                adjusted_p=float(row["adjusted_p_value"]),
                effect=format_effect(row["effect_size_dz"]),
                interpretation=row["interpretation"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def result_sentence(row: dict[str, object]) -> str:
    prefix = (
        f"On {row['dataset_display']} for {row['metric_display']}, QRSAN differed from "
        f"{row['comparator_display']} by {format_pp(float(row['delta']))} percentage points "
        f"(95% CI, {format_pp(float(row['ci_lower']))} to {format_pp(float(row['ci_upper']))}; "
        f"Holm-adjusted p = {float(row['adjusted_p_value']):.4g}; "
        f"paired Cohen's d_z = {format_effect(row['effect_size_dz'])})."
    )
    if row["interpretation"] == "Superior":
        return prefix + " QRSAN significantly exceeded the comparator."
    if row["interpretation"] == "Inferior":
        return prefix + " QRSAN was significantly lower than the comparator."
    return prefix + " The difference was not statistically supported, so QRSAN is described as competitive."


def write_section_text(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# Section 4.3 replacement text",
        "",
        "To assess whether the observed performance differences were robust, we retained the test-set "
        f"predictions from {len(args.batch_ids)} matched random-seed runs for QRSAN and the "
        "pre-specified reproduced comparator on each dataset. For each metric, we estimated "
        "the paired performance difference, (Δ = QRSAN − baseline), using a paired bootstrap procedure "
        f"with {args.bootstrap_resamples:,} resamples of the identical test instances. We calculated "
        f"two-sided paired permutation p-values with {args.permutation_resamples:,} prediction-swap "
        "resamples and adjusted all p-values across the dataset-metric family using the Holm procedure. "
        f"We report the mean difference, 95% confidence interval, adjusted p-value, and paired Cohen's d_z "
        f"in Table {args.table_label}. Superiority language is used only when the adjusted p-value is below "
        "0.05 and the corresponding confidence interval excludes zero. Otherwise, results are described as competitive.",
        "",
    ]
    lines.extend(result_sentence(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reviewer_snippet(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# Reviewer-response snippet",
        "",
        "We thank the reviewer for this important suggestion. We retained test-set predictions from "
        f"{len(args.batch_ids)} matched random-seed runs and conducted paired bootstrap comparisons "
        "and two-sided paired permutation tests between QRSAN and the pre-specified reproduced "
        f"comparator for each dataset and metric. We now provide the mean difference, 95% confidence interval, "
        f"Holm-adjusted p-value, and paired Cohen's d_z in Table {args.table_label}. We use superiority "
        "language only where the statistical comparison supports it; otherwise, we describe QRSAN as competitive.",
        "",
    ]
    lines.extend(result_sentence(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    comparators = parse_comparators(args.comparators, args.datasets)
    input_root = Path(args.input_root)
    paired_by_dataset: dict[str, list[PairedRun]] = {}
    for dataset in args.datasets:
        comparator = comparators[dataset]
        runs: list[PairedRun] = []
        for batch_id in args.batch_ids:
            qrsan_path = prediction_path(input_root, batch_id, dataset, "qrsan")
            baseline_path = prediction_path(input_root, batch_id, dataset, comparator)
            missing = [str(path) for path in (qrsan_path, baseline_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError("missing prediction file(s): " + ", ".join(missing))
            runs.append(PairedRun(
                run_id=run_id(batch_id),
                qrsan=load_run(qrsan_path, dataset),
                baseline=load_run(baseline_path, dataset),
            ))
        verify_paired_runs(runs)
        paired_by_dataset[dataset] = runs

    seed_sequence = np.random.SeedSequence(args.random_seed)
    bootstrap_seeds, permutation_seeds = seed_sequence.spawn(2)
    bootstrap_generators = [np.random.default_rng(seed) for seed in bootstrap_seeds.spawn(32)]
    permutation_generators = [np.random.default_rng(seed) for seed in permutation_seeds.spawn(32)]
    statistics_rows: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []
    generator_index = 0
    for dataset in args.datasets:
        comparator = comparators[dataset]
        compatible_metrics = IEMOCAP_METRICS if dataset == "iemocap" else SENTIMENT_METRICS
        selected_metrics = [metric for metric in args.metrics if metric in compatible_metrics]
        skipped_metrics = sorted(set(args.metrics) - set(selected_metrics))
        if skipped_metrics:
            print(
                f"Skipping unsupported metric(s) for {dataset}: {', '.join(skipped_metrics)}",
                flush=True,
            )
        runs = paired_by_dataset[dataset]
        for metric in selected_metrics:
            qrsan_scores = [
                metric_value(metric, run.qrsan.prediction, run.qrsan.target, dataset)
                for run in runs
            ]
            baseline_scores = [
                metric_value(metric, run.baseline.prediction, run.baseline.target, dataset)
                for run in runs
            ]
            paired_differences = [qrsan - baseline for qrsan, baseline in zip(qrsan_scores, baseline_scores)]
            bootstrap_distribution = paired_bootstrap(
                runs, dataset, metric, args.bootstrap_resamples,
                bootstrap_generators[generator_index],
            )
            permutation_distribution = paired_permutation(
                runs, dataset, metric, args.permutation_resamples,
                permutation_generators[generator_index],
            )
            generator_index += 1
            qrsan_mean, qrsan_std = mean_and_std(qrsan_scores)
            baseline_mean, baseline_std = mean_and_std(baseline_scores)
            observed_difference = float(np.mean(paired_differences))
            unadjusted_p = (1 + int(np.count_nonzero(
                np.abs(permutation_distribution) >= abs(observed_difference) - 1e-15,
            ))) / (args.permutation_resamples + 1)
            row: dict[str, object] = {
                "dataset": dataset,
                "dataset_display": DISPLAY_DATASET[dataset],
                "metric": metric,
                "metric_display": DISPLAY_METRIC[metric],
                "comparator": comparator,
                "comparator_display": DISPLAY_MODEL.get(comparator, comparator),
                "matched_runs": len(runs),
                "test_instances": runs[0].qrsan.n_instances,
                "qrsan_mean": qrsan_mean,
                "qrsan_std": qrsan_std,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "delta": observed_difference,
                "ci_lower": float(np.quantile(bootstrap_distribution, 0.025)),
                "ci_upper": float(np.quantile(bootstrap_distribution, 0.975)),
                "p_value": float(unadjusted_p),
                "effect_size_dz": cohen_dz(paired_differences),
                "bootstrap_resamples": args.bootstrap_resamples,
                "permutation_resamples": args.permutation_resamples,
            }
            statistics_rows.append(row)
            for run, qrsan_score, baseline_score, difference in zip(
                runs, qrsan_scores, baseline_scores, paired_differences,
            ):
                per_seed_rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "comparator": comparator,
                    "run": run.run_id,
                    "qrsan": qrsan_score,
                    "baseline": baseline_score,
                    "delta": difference,
                    "qrsan_prediction_file": str(run.qrsan.source),
                    "baseline_prediction_file": str(run.baseline.source),
                })

    if not statistics_rows:
        raise ValueError("no compatible dataset/metric comparisons were selected")
    holm_adjust(statistics_rows)
    for row in statistics_rows:
        row["interpretation"] = interpretation(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "per_seed_metrics.csv", per_seed_rows,
        [
            "dataset", "metric", "comparator", "run", "qrsan", "baseline", "delta",
            "qrsan_prediction_file", "baseline_prediction_file",
        ],
    )
    write_csv(
        output_dir / "paired_statistical_comparisons.csv", statistics_rows,
        [
            "dataset", "metric", "comparator", "matched_runs", "test_instances",
            "qrsan_mean", "qrsan_std", "baseline_mean", "baseline_std", "delta",
            "ci_lower", "ci_upper", "p_value", "adjusted_p_value", "effect_size_dz",
            "bootstrap_resamples", "permutation_resamples", "interpretation",
        ],
    )
    write_markdown_table(
        output_dir / "paired_statistical_comparisons.md", statistics_rows, args.table_label,
    )
    write_section_text(output_dir / "section_4_3_replacement.md", statistics_rows, args)
    write_reviewer_snippet(output_dir / "reviewer_response_snippet.md", statistics_rows, args)
    metadata = {
        "input_root": str(input_root),
        "batch_ids": args.batch_ids,
        "datasets": args.datasets,
        "comparators": comparators,
        "metrics_requested": args.metrics,
        "bootstrap_resamples": args.bootstrap_resamples,
        "permutation_resamples": args.permutation_resamples,
        "random_seed": args.random_seed,
        "p_value_adjustment": "Holm across all rows in paired_statistical_comparisons.csv",
        "confidence_interval": "2.5th to 97.5th percentile paired bootstrap interval",
        "effect_size": "paired Cohen's dz over matched seed-level metric differences",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Wrote paired statistical comparison outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
