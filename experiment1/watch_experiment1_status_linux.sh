#!/usr/bin/env bash
# Read-only live status view for Experiment 1.  Ctrl-C stops only this viewer.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV=""
INTERVAL=15
while (($#)); do
  case "$1" in
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    -h|--help) echo "Usage: bash experiment1/watch_experiment1_status_linux.sh [--conda-env ENV] [--interval SECONDS]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
cd "$ROOT"
if [[ -n "$CONDA_ENV" ]]; then RUNNER=(conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN"); else RUNNER=("$PYTHON_BIN"); fi
watch -n "$INTERVAL" "${RUNNER[*]} experiment1/status_experiment1.py"
