"""Report class-imbalance baselines from saved Section 4.3 predictions.

Supports binary CMU-MOSEI/CMU-MOSI sentiment and the four-label IEMOCAP
emotion task.  Majority labels are determined solely from the training split.
For IEMOCAP, majority and uniform-random predictions are produced separately
for each emotion label, and balanced accuracy/macro-F1 are macro-averaged over
the four binary label tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
DISPLAY_DATASET = {
    "cmumosei": "CMU-MOSEI",
    "cmumosi": "CMU-MOSI",
    "iemocap": "IEMOCAP",
}


def run_name(path: Path) -> str:
    for part in reversed(path.parts):
        if re.fullmatch(r"matrix_[A-Za-z0-9_.-]+", part):
            return part
    return path.stem.removesuffix(".predictions")


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        missing = {"outputs", "targets"} - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}")
        return np.asarray(archive["outputs"]), np.asarray(archive["targets"])


def load_train_values(path: Path, label_key: str) -> tuple[np.ndarray, str]:
    with path.open("rb") as stream:
        data = pickle.load(stream)
    try:
        train = data["train"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} does not contain a train split") from error
    resolved = label_key
    # Prepared data variants use either the semantic task name (``sentiment``
    # or ``emotion``) or the generic ``labels`` key.
    if resolved not in train and label_key in {"sentiment", "emotion"} and "labels" in train:
        resolved = "labels"
    if resolved not in train:
        raise ValueError(
            f"{path} lacks train/{label_key}; available keys: {', '.join(sorted(train))}"
        )
    return np.asarray(train[resolved]), resolved


def sentiment_labels(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1) >= 0


def iemocap_labels(values: np.ndarray, source: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 8:
        values = values.reshape(-1, 4, 2)
    if values.ndim == 3 and values.shape[1:] == (4, 2):
        return values.argmax(axis=-1).astype(bool)
    if values.ndim == 2 and values.shape[1] == 4:
        return values.astype(bool)
    raise ValueError(f"{source}: expected IEMOCAP labels shaped (N, 4) or (N, 4, 2), got {values.shape}")


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=bool).reshape(-1)
    prediction = np.asarray(prediction, dtype=bool).reshape(-1)
    if target.size == 0 or target.shape != prediction.shape:
        raise ValueError("invalid target/prediction arrays")
    tn = int(np.count_nonzero(~target & ~prediction))
    fp = int(np.count_nonzero(~target & prediction))
    fn = int(np.count_nonzero(target & ~prediction))
    tp = int(np.count_nonzero(target & prediction))
    neg_recall = tn / (tn + fp) if tn + fp else 0.0
    pos_recall = tp / (tp + fn) if tp + fn else 0.0
    neg_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    pos_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "accuracy": (tn + tp) / target.size,
        "balanced_accuracy": (neg_recall + pos_recall) / 2,
        "macro_f1": (neg_f1 + pos_f1) / 2,
    }


def score(task: str, target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if task == "sentiment":
        return binary_metrics(target, prediction)
    per_label = [binary_metrics(target[:, index], prediction[:, index]) for index in range(4)]
    return {metric: float(np.mean([row[metric] for row in per_label])) for metric in METRICS}


def summary(method: str, source: str, rows: list[dict[str, object]], majority: str = "") -> dict[str, object]:
    result: dict[str, object] = {
        "method": method, "source": source, "majority_class": majority,
        "n_evaluations": len(rows),
    }
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows])
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_mean_std(row: dict[str, object], metric: str) -> str:
    return f"{100 * float(row[f'{metric}_mean']):.2f} ± {100 * float(row[f'{metric}_std']):.2f}"


def write_markdown(path: Path, dataset: str, task: str, model: str, summaries: list[dict[str, object]], n_train: int, n_test: int, draws: int) -> None:
    if task == "sentiment":
        protocol = "Binary labels use sentiment score ≥ 0 as positive."
    else:
        protocol = "IEMOCAP metrics are macro-averaged across its four binary emotion labels."
    lines = [
        f"# Section 4.3 class-imbalance robustness: {DISPLAY_DATASET[dataset]}", "",
        f"Protocol. {model} and both references were evaluated on the same {n_test} saved test instances. {protocol} "
        f"Majority labels were selected only from {n_train} training instances. Uniform-random results are mean ± sample standard deviation over {draws} deterministic draws.", "",
        "| Method | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        name = str(row["method"])
        if row["majority_class"]:
            name += f" ({row['majority_class']} from training split)"
        lines.append("| {} | {} | {} | {} |".format(
            name, *(format_mean_std(row, metric) for metric in METRICS),
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DISPLAY_DATASET), required=True)
    parser.add_argument("--dataset-pickle", required=True)
    parser.add_argument("--prediction-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="QRSAN")
    parser.add_argument("--label-key")
    parser.add_argument("--uniform-random-draws", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.uniform_random_draws < 2:
        parser.error("--uniform-random-draws must be at least 2")
    if args.label_key is None:
        args.label_key = "emotion" if args.dataset == "iemocap" else "sentiment"
    return args


def main() -> int:
    args = parse_args()
    task = "iemocap" if args.dataset == "iemocap" else "sentiment"
    files = [Path(value) for value in args.prediction_files]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing prediction file(s): " + ", ".join(missing))
    train_raw, label_key = load_train_values(Path(args.dataset_pickle), args.label_key)
    train = iemocap_labels(train_raw, "training labels") if task == "iemocap" else sentiment_labels(train_raw)
    majority = train.mean(axis=0) > 0.5
    majority_text = ", ".join("positive" if value else "negative" for value in np.asarray(majority).reshape(-1))
    model_rows: list[dict[str, object]] = []
    test: np.ndarray | None = None
    for path in files:
        outputs, targets = load_prediction(path)
        target = iemocap_labels(targets, str(path)) if task == "iemocap" else sentiment_labels(targets)
        prediction = iemocap_labels(outputs, str(path)) if task == "iemocap" else sentiment_labels(outputs)
        if test is None:
            test = target
        elif not np.array_equal(test, target):
            raise ValueError(f"{path}: test targets differ from the first saved run")
        row: dict[str, object] = {"method": args.model_name, "kind": "model", "source": str(path), "run": run_name(path)}
        row.update(score(task, target, prediction))
        model_rows.append(row)
    assert test is not None
    majority_prediction = np.broadcast_to(majority, test.shape).copy()
    majority_row: dict[str, object] = {"method": "Majority-class baseline", "kind": "majority_baseline", "source": "training split", "run": "fixed"}
    majority_row.update(score(task, test, majority_prediction))
    generator = np.random.default_rng(args.random_seed)
    random_rows: list[dict[str, object]] = []
    for draw in range(args.uniform_random_draws):
        prediction = generator.integers(0, 2, size=test.shape, dtype=np.int8).astype(bool)
        row = {"method": "Uniform-random baseline", "kind": "uniform_random_baseline", "source": f"seed={args.random_seed}", "run": f"draw_{draw + 1:04d}"}
        row.update(score(task, test, prediction))
        random_rows.append(row)
    summaries = [
        summary(args.model_name, "saved predictions", model_rows),
        summary("Majority-class baseline", "training split", [majority_row], majority_text),
        summary("Uniform-random baseline", f"seed={args.random_seed}", random_rows),
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_evaluation_metrics.csv", [*model_rows, majority_row, *random_rows], ["method", "kind", "source", "run", *METRICS])
    write_csv(output / "summary_mean_std.csv", summaries, ["method", "source", "majority_class", "n_evaluations", *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")]])
    write_markdown(output / "section_4_3_table.md", args.dataset, task, args.model_name, summaries, int(train.shape[0]), int(test.shape[0]), args.uniform_random_draws)
    (output / "metadata.json").write_text(json.dumps({"dataset": args.dataset, "task": task, "training_label_key": label_key, "prediction_files": [str(path) for path in files], "uniform_random_draws": args.uniform_random_draws, "uniform_random_seed": args.random_seed, "majority_labels": np.asarray(majority, dtype=int).reshape(-1).tolist()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DISPLAY_DATASET[args.dataset]} robustness table to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
