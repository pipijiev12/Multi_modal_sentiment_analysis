"""Create reproducible class-level Figure 4 metrics from saved predictions.

Each input .npz must have ``outputs`` and ``targets`` arrays, as written by
``utils.model.test``.  The script intentionally refuses metrics-only CSV files:
per-class precision and recall cannot be recovered from aggregate scores.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


SENTIMENT_CLASSES = (("Negative", 0), ("Positive", 1))
IEMOCAP_CLASSES = ("Neutral", "Happy", "Sad", "Angry")
METRICS = ("precision", "recall", "f1")


def prediction_files(inputs: list[str]) -> list[Path]:
    """Expand input files/directories, while failing loudly for CSV-only input."""
    files: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.npz")))
        elif path.suffix.lower() == ".npz" and path.is_file():
            files.append(path)
        elif path.exists():
            raise ValueError(f"{path} is not a .npz prediction file")
        else:
            raise FileNotFoundError(path)
    files = sorted(set(files))
    if not files:
        raise ValueError("No .npz prediction files were found.")
    return files


def run_name(path: Path) -> str:
    """Use matrix/run directory identity so the five runs remain traceable."""
    for part in reversed(path.parts):
        if re.fullmatch(r"(?:matrix_)?[A-Za-z0-9_.-]*\d+[A-Za-z0-9_.-]*", part):
            return part
    return path.stem


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        missing = {"outputs", "targets"} - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(sorted(missing))}")
        return np.asarray(archive["outputs"]), np.asarray(archive["targets"])


def score_binary(y_true: np.ndarray, y_pred: np.ndarray, class_name: str, source: Path) -> tuple[dict[str, object], dict[str, object]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0,
    )
    rows = []
    for name, index in SENTIMENT_CLASSES:
        rows.append({
            "run": run_name(source), "source": str(source), "class": name,
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
        })
    matrix = {
        "run": run_name(source), "source": str(source), "class": class_name,
        "true_negative_pred_negative": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 0]),
        "true_negative_pred_positive": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 1]),
        "true_positive_pred_negative": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 0]),
        "true_positive_pred_positive": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 1]),
    }
    return rows, matrix


def score_sentiment(outputs: np.ndarray, targets: np.ndarray, source: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y_pred = (outputs.reshape(-1) >= 0).astype(int)
    y_true = (targets.reshape(-1) >= 0).astype(int)
    rows, matrix = score_binary(y_true, y_pred, "Binary sentiment", source)
    return rows, [matrix]


def score_iemocap(outputs: np.ndarray, targets: np.ndarray, source: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    # ``utils.model.test`` reshapes model outputs to the target shape for
    # evaluation, but stores the raw model output.  QRSAN therefore writes
    # (N, 8) logits while IEMOCAP targets remain (N, 4, 2).
    if outputs.ndim == 2 and outputs.shape[1] == 8:
        outputs = outputs.reshape(-1, 4, 2)
    if targets.ndim == 2 and targets.shape[1] == 8:
        targets = targets.reshape(-1, 4, 2)
    if outputs.ndim < 3 or outputs.shape[-2:] != (4, 2) or targets.shape != outputs.shape:
        raise ValueError(
            f"{source}: expected IEMOCAP arrays shaped (samples, 4, 2), got "
            f"outputs={outputs.shape}, targets={targets.shape}"
        )
    predicted = outputs.reshape(-1, 4, 2).argmax(axis=-1)
    actual = targets.reshape(-1, 4, 2).argmax(axis=-1)
    rows: list[dict[str, object]] = []
    matrices: list[dict[str, object]] = []
    for index, emotion in enumerate(IEMOCAP_CLASSES):
        precision, recall, f1, support = precision_recall_fscore_support(
            actual[:, index], predicted[:, index], labels=[1], zero_division=0,
        )
        cm = confusion_matrix(actual[:, index], predicted[:, index], labels=[0, 1])
        rows.append({
            "run": run_name(source), "source": str(source), "class": emotion,
            "precision": float(precision[0]), "recall": float(recall[0]),
            "f1": float(f1[0]), "support": int(support[0]),
        })
        matrices.append({
            "run": run_name(source), "source": str(source), "class": emotion,
            "true_negative_pred_negative": int(cm[0, 0]),
            "true_negative_pred_positive": int(cm[0, 1]),
            "true_positive_pred_negative": int(cm[1, 0]),
            "true_positive_pred_positive": int(cm[1, 1]),
        })
    return rows, matrices


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for class_name in dict.fromkeys(row["class"] for row in rows):
        selected = [row for row in rows if row["class"] == class_name]
        result: dict[str, object] = {"class": class_name, "runs": len(selected)}
        for metric in METRICS:
            values = np.array([float(row[metric]) for row in selected])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def plot(dataset: str, summary: list[dict[str, object]], output_path: Path) -> None:
    class_names = [str(row["class"]) for row in summary]
    positions = np.arange(len(class_names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(7, len(class_names) * 1.7), 4.8))
    for offset, metric in enumerate(METRICS):
        means = [float(row[f"{metric}_mean"]) for row in summary]
        stds = [float(row[f"{metric}_std"]) for row in summary]
        axis.bar(positions + (offset - 1) * width, means, width, yerr=stds,
                 capsize=4, label=metric.capitalize())
    axis.set_xticks(positions, class_names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(f"Figure 4: {dataset} class-level metrics (mean ± std)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cmumosi", "cmumosei", "iemocap"))
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Prediction .npz files or directories containing them.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    files = prediction_files(args.inputs)
    rows: list[dict[str, object]] = []
    matrices: list[dict[str, object]] = []
    for path in files:
        outputs, targets = load_arrays(path)
        if args.dataset == "iemocap":
            run_rows, run_matrices = score_iemocap(outputs, targets, path)
        else:
            run_rows, run_matrices = score_sentiment(outputs, targets, path)
        rows.extend(run_rows)
        matrices.extend(run_matrices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(rows)
    write_csv(output_dir / "per_run_class_metrics.csv", rows,
              ["run", "source", "class", "precision", "recall", "f1", "support"])
    write_csv(output_dir / "summary_mean_std.csv", summary,
              ["class", "runs", "precision_mean", "precision_std", "recall_mean", "recall_std", "f1_mean", "f1_std"])
    write_csv(output_dir / "confusion_matrices.csv", matrices,
              ["run", "source", "class", "true_negative_pred_negative", "true_negative_pred_positive", "true_positive_pred_negative", "true_positive_pred_positive"])
    plot(args.dataset, summary, output_dir / f"figure4_{args.dataset}.png")
    print(f"Processed {len(files)} prediction files -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
