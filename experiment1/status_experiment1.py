"""Report completed, running, incomplete and pending Experiment 1 jobs.

The report is read-only.  A job is complete only when both metrics.csv and
predictions.npz exist.  Running jobs are identified from the live ``run.py
-config .../experiment1/generated/...`` command line, so interrupted jobs are
not mistakenly labelled as running.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def process_lines() -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return []
    return result.stdout.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", help="Limit output to one or more suite names.")
    parser.add_argument("--details", action="store_true", help="Print every running or incomplete job.")
    args = parser.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(args.suite or manifest["experiments"].keys())
    unknown = wanted - set(manifest["experiments"])
    if unknown:
        parser.error("unknown suite(s): " + ", ".join(sorted(unknown)))

    commands = process_lines()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    details: dict[str, list[str]] = defaultdict(list)
    overall: Counter[str] = Counter()

    for suite in manifest["experiments"]:
        if suite not in wanted or suite == "basis_sensitivity":
            continue
        specs = [item if isinstance(item, dict) else {"name": item} for item in manifest["experiments"][suite]]
        for seed in manifest["seeds"]:
            for dataset in manifest["datasets"]:
                for spec in specs:
                    model = spec["name"]
                    run_dir = ROOT / "eval" / "experiment1" / suite / f"seed_{seed}" / dataset / model
                    complete = (run_dir / "metrics.csv").is_file() and (run_dir / "predictions.npz").is_file()
                    config_suffix = f"experiment1/generated/{suite}/seed_{seed}/{dataset}/{model}.ini"
                    active = any(config_suffix in line.replace("\\", "/") for line in commands)
                    if complete:
                        state = "completed"
                    elif active:
                        state = "running"
                    elif run_dir.exists():
                        state = "incomplete"
                    else:
                        state = "pending"
                    counts[suite][state] += 1
                    overall[state] += 1
                    if args.details and state in {"running", "incomplete"}:
                        details[state].append(f"{suite}/seed_{seed}/{dataset}/{model}")

    fields = ("completed", "running", "incomplete", "pending")
    headers = ("suite", *fields, "total")
    rows = [
        (suite, *(counts[suite][field] for field in fields), sum(counts[suite].values()))
        for suite in sorted(counts)
    ]
    rows.append(("TOTAL", *(overall[field] for field in fields), sum(overall.values())))
    widths = [max(len(str(header)), *(len(str(row[index])) for row in rows)) for index, header in enumerate(headers)]
    def render(row: tuple[object, ...]) -> str:
        return "  ".join(
            str(value).ljust(widths[index]) if index == 0 else str(value).rjust(widths[index])
            for index, value in enumerate(row)
        )
    print(render(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(render(row))
    print("\ncompleted: metrics.csv + predictions.npz; incomplete: created but neither completed nor live.")
    if args.details:
        for state in ("running", "incomplete"):
            if details[state]:
                print(f"\n{state.upper()} ({len(details[state])}):")
                print("\n".join(details[state]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
