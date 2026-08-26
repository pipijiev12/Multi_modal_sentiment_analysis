#!/usr/bin/env bash
# Read-only snapshot of current server hardware and active GPU processes.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="multimodal-sa"
OUTPUT_DIR="eval/experiment1/audits/hardware"
while (($#)); do
  case "$1" in
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) echo "Usage: bash experiment1/collect_hardware_environment_linux.sh [--conda-env ENV] [--output-dir DIR]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
cd "$ROOT"
conda run --no-capture-output -n "$CONDA_ENV" python3 experiment1/collect_hardware_environment.py --output-dir "$OUTPUT_DIR"
