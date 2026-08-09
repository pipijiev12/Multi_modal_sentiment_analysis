#!/usr/bin/env bash
# Run the missing Figure 5 / Figure 6 repeated experiments on two GPUs.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV=""
GPUS=(0 1)
FIGURE="all"
EPOCHS=""
BASE_SEED=2024
NO_RESUME=0
CONTINUE_ON_ERROR=1
DRY_RUN=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/train_figures_linux.sh [options]

Default workload: Figure 5 (90 missing-modality QRSAN runs) + Figure 6
(15 Basic QDNN + 15 QSAN + 15 residual QRSAN runs) = 135
experiments, distributed across GPU 0 and GPU 1.

Options:
  --figure NAME            figure5, figure6, or all (default: all)
  --gpus LIST              Comma-separated GPU IDs (default: 0,1)
  --epochs N               Override epoch count
  --base-seed N            First of five seeds (default: 2024; uses 2024-2028)
  --python PATH            Python executable (default: python3)
  --conda-env NAME         Use conda run -n NAME
  --fresh                  Ignore existing metrics.csv files
  --check                  Run one forward/backward batch per experiment
  --stop-on-error          Stop an assigned worker after its first failure
  --dry-run                Print assigned jobs without training
  -h, --help               Show this help
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

split_csv() {
  local value="$1"
  local -n destination="$2"
  IFS=',' read -r -a destination <<<"${value}"
  ((${#destination[@]} > 0)) || die "empty GPU list"
}

while (($# > 0)); do
  case "$1" in
    --figure) FIGURE="$2"; shift 2 ;;
    --gpus) split_csv "$2" GPUS; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --base-seed) BASE_SEED="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --fresh) NO_RESUME=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    --stop-on-error) CONTINUE_ON_ERROR=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "${FIGURE}" =~ ^(figure5|figure6|all)$ ]] || die "invalid --figure: ${FIGURE}"
[[ "${BASE_SEED}" =~ ^[0-9]+$ ]] || die "--base-seed must be a non-negative integer"
[[ -z "${EPOCHS}" || "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be positive"
for gpu_id in "${GPUS[@]}"; do [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${gpu_id}"; done

if [[ -n "${CONDA_ENV}" ]]; then
  command -v conda >/dev/null 2>&1 || die "conda was not found in PATH"
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}")
else
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
  RUNNER=("${PYTHON_BIN}")
fi

run_worker() {
  local worker_index="$1"
  local gpu_id="${GPUS[worker_index]}"
  local command=("${RUNNER[@]}" -u scripts/train_figures.py
    --figure "${FIGURE}" --runs 5 --base-seed "${BASE_SEED}"
    --worker-index "${worker_index}" --worker-count "${#GPUS[@]}")
  [[ -z "${EPOCHS}" ]] || command+=(--epochs "${EPOCHS}")
  ((NO_RESUME == 0)) || command+=(--no-resume)
  ((CONTINUE_ON_ERROR == 0)) || command+=(--continue-on-error)
  ((CHECK_ONLY == 0)) || command+=(--check)
  ((DRY_RUN == 0)) || command+=(--dry-run)
  printf '[worker %s] GPU=%s: ' "${worker_index}" "${gpu_id}"
  printf '%q ' env "CUDA_VISIBLE_DEVICES=${gpu_id}" "${command[@]}"
  printf '\n'
  ((DRY_RUN == 1)) && return 0
  mkdir -p "${PROJECT_ROOT}/eval/${FIGURE}"
  env CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}" 2>&1 | tee "${PROJECT_ROOT}/eval/${FIGURE}/worker_${worker_index}.log"
}

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
PIDS=()
for ((worker=0; worker<${#GPUS[@]}; worker++)); do
  run_worker "${worker}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
