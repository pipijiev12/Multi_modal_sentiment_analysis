"""Create the binary-baseline robustness table required for Section 4.3.

The script evaluates every method on exactly the test labels stored next to its
predictions.  The majority-class label is selected only from the training
split, so it cannot exploit the test-set class distribution.  The uniform
random reference is a deterministic Monte Carlo estimate, controlled by
``--random-seed``.

Example
-------
python scripts/report_binary_robustness.py ^
  --dataset-pickle data/cmumosi_cmumosei_iemocap_mult/cmumosei_data.pkl ^
  --prediction-files eval/matrix_1/cmumosei/qrsan.predictions.npz ^
                     eval/matrix_2/cmumosei/qrsan.predictions.npz ^
                     eval/matrix_3/cmumosei/qrsan.predictions.npz ^
                     eval/matrix_4/cmumosei/qrsan.predictions.npz ^
                     eval/matrix_5/cmumosei/qrsan.predictions.npz ^
  --model-name QRSAN --output-dir eval/section_4_3/cmumosei_qrsan
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
POSITIVE_THRESHOLD = 0.0


def binary_labels(values: object) -> np.ndarray:
    """Apply the repository's sentiment convention: score >= 0 is positive."""
    return np.asarray(values).reshape(-1) >= POSITIVE_THRESHOLD


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate fixed-label binary metrics without silently dropping a class."""
    y_true = np.asarray(y_true, dtype=bool).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=bool).reshape(-1)
    if y_true.size == 0:
        raise ValueError("cannot score an empty evaluation set")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"target/prediction size mismatch: {y_true.shape} != {y_pred.shape}"
        )

    true_negative = int((~y_true & ~y_pred).sum())
    false_positive = int((~y_true & y_pred).sum())
    false_negative = int((y_true & ~y_pred).sum())
    true_positive = int((y_true & y_pred).sum())

    negative_recall = true_negative / (true_negative + false_positive) \
        if true_negative + false_positive else 0.0
    positive_recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    negative_f1 = (2 * true_negative) / (
        2 * true_negative + false_positive + false_negative
    ) if 2 * true_negative + false_positive + false_negative else 0.0
    positive_f1 = (2 * true_positive) / (
        2 * true_positive + false_positive + false_negative
    ) if 2 * true_positive + false_positive + false_negative else 0.0

    return {
        "accuracy": (true_negative + true_positive) / y_true.size,
        "balanced_accuracy": (negative_recall + positive_recall) / 2,
        "macro_f1": (negative_f1 + positive_f1) / 2,
    }


def run_name(path: Path) -> str:
    """Keep a stable, human-readable identifier for each repeated run."""
    for part in reversed(path.parts):
        if re.fullmatch(r"matrix_[A-Za-z0-9_.-]+", part):
            return part
    return path.stem.removesuffix(".predictions")


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        required = {"outputs", "targets"}
        missing = required - set(archive.files)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing array(s): {missing_text}")
        return np.asarray(archive["outputs"]), np.asarray(archive["targets"])


def load_training_targets(path: Path, label_key: str) -> tuple[np.ndarray, str]:
    with path.open("rb") as stream:
        dataset = pickle.load(stream)
    try:
        train_split = dataset["train"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} does not contain a train split") from error
    # Older prepared datasets use ``labels`` whereas some source variants use
    # ``sentiment``.  The automatic fallback retains the experiment's default
    # while making the report work with both repository data layouts.
    resolved_label_key = label_key
    if resolved_label_key not in train_split and label_key == "sentiment":
        resolved_label_key = "labels"
    try:
        values = train_split[resolved_label_key]
    except KeyError as error:
        available = ", ".join(sorted(train_split))
        raise ValueError(
            f"{path} does not contain train/{label_key} labels "
            f"(available keys: {available})"
        ) from error
    return binary_labels(values), resolved_label_key


def summary_row(
    method: str, source: str, rows: Iterable[dict[str, object]],
    majority_class: str = "",
) -> dict[str, object]:
    selected = list(rows)
    if not selected:
        raise ValueError(f"no rows to summarise for {method}")
    summary: dict[str, object] = {
        "method": method,
        "source": source,
        "majority_class": majority_class,
        "n_evaluations": len(selected),
    }
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in selected])
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = (
            float(values.std(ddof=1)) if values.size > 1 else 0.0
        )
    return summary


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_mean_std(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def write_markdown_table(
    path: Path, summaries: list[dict[str, object]], model_name: str,
    n_test_samples: int, n_train_samples: int, random_draws: int,
) -> None:
    lines = [
        "# Section 4.3 binary robustness table",
        "",
        f"Protocol. {model_name} and both reference classifiers were evaluated on "
        f"the same {n_test_samples} saved CMU-MOSEI test predictions. Binary "
        "labels use sentiment score >= 0 as positive. The majority class was "
        f"selected from the {n_train_samples} training labels only. Uniform-random "
        f"results are mean ± sample standard deviation over {random_draws} "
        "deterministic Bernoulli(0.5) draws.",
        "",
        "| Method | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        name = str(row["method"])
        if row["majority_class"]:
            name = f"{name} ({row['majority_class']} from training split)"
        lines.append(
            "| {name} | {accuracy} | {balanced_accuracy} | {macro_f1} |".format(
                name=name,
                accuracy=format_mean_std(
                    float(row["accuracy_mean"]), float(row["accuracy_std"]),
                ),
                balanced_accuracy=format_mean_std(
                    float(row["balanced_accuracy_mean"]),
                    float(row["balanced_accuracy_std"]),
                ),
                macro_f1=format_mean_std(
                    float(row["macro_f1_mean"]), float(row["macro_f1_std"]),
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-pickle", required=True,
        help="Dataset pickle containing train/sentiment labels.",
    )
    parser.add_argument(
        "--prediction-files", nargs="+", required=True,
        help="One saved test prediction .npz file per model seed.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Model")
    parser.add_argument("--label-key", default="sentiment")
    parser.add_argument("--uniform-random-draws", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.uniform_random_draws < 2:
        parser.error("--uniform-random-draws must be at least 2")
    return args


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_pickle)
    prediction_paths = [Path(value) for value in args.prediction_files]
    missing = [str(path) for path in prediction_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing prediction file(s): " + ", ".join(missing))

    train_labels, resolved_label_key = load_training_targets(
        dataset_path, args.label_key,
    )
    if train_labels.size == 0:
        raise ValueError("the training split contains no labels")
    positive_train = int(train_labels.sum())
    negative_train = int(train_labels.size - positive_train)
    majority_is_positive = positive_train > negative_train
    majority_class = "positive" if majority_is_positive else "negative"

    model_rows: list[dict[str, object]] = []
    test_labels: np.ndarray | None = None
    for path in prediction_paths:
        outputs, targets = load_prediction(path)
        target_labels = binary_labels(targets)
        if test_labels is None:
            test_labels = target_labels
        elif not np.array_equal(test_labels, target_labels):
            raise ValueError(
                "all prediction files must contain the same ordered test targets; "
                f"{path} differs from the first file"
            )
        row: dict[str, object] = {
            "method": args.model_name,
            "kind": "model",
            "source": str(path),
            "run": run_name(path),
        }
        row.update(binary_metrics(target_labels, binary_labels(outputs)))
        model_rows.append(row)

    assert test_labels is not None  # argparse requires at least one path
    majority_predictions = np.full(
        test_labels.shape, majority_is_positive, dtype=bool,
    )
    majority_row: dict[str, object] = {
        "method": "Majority-class baseline",
        "kind": "majority_baseline",
        "source": "training split",
        "run": "fixed",
    }
    majority_row.update(binary_metrics(test_labels, majority_predictions))

    generator = np.random.default_rng(args.random_seed)
    uniform_rows: list[dict[str, object]] = []
    for draw_index in range(args.uniform_random_draws):
        random_predictions = generator.integers(
            0, 2, size=test_labels.size, dtype=np.int8,
        ).astype(bool)
        row = {
            "method": "Uniform-random baseline",
            "kind": "uniform_random_baseline",
            "source": f"seed={args.random_seed}",
            "run": f"draw_{draw_index + 1:04d}",
        }
        row.update(binary_metrics(test_labels, random_predictions))
        uniform_rows.append(row)

    summaries = [
        summary_row(args.model_name, "saved predictions", model_rows),
        summary_row(
            "Majority-class baseline", "training split", [majority_row],
            majority_class=majority_class,
        ),
        summary_row(
            "Uniform-random baseline", f"seed={args.random_seed}", uniform_rows,
        ),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "per_evaluation_metrics.csv",
        [*model_rows, majority_row, *uniform_rows],
        ["method", "kind", "source", "run", *METRICS],
    )
    write_csv(
        output_dir / "summary_mean_std.csv", summaries,
        [
            "method", "source", "majority_class", "n_evaluations",
            *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")],
        ],
    )
    write_markdown_table(
        output_dir / "section_4_3_table.md", summaries, args.model_name,
        int(test_labels.size), int(train_labels.size), args.uniform_random_draws,
    )
    metadata = {
        "binary_label_rule": "sentiment score >= 0 is positive",
        "dataset_pickle": str(dataset_path),
        "training_label_key": resolved_label_key,
        "prediction_files": [str(path) for path in prediction_paths],
        "n_train_labels": int(train_labels.size),
        "n_test_labels": int(test_labels.size),
        "training_positive_labels": positive_train,
        "training_negative_labels": negative_train,
        "majority_class": majority_class,
        "uniform_random_draws": args.uniform_random_draws,
        "uniform_random_seed": args.random_seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Wrote Section 4.3 robustness table to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
