#!/usr/bin/env bash
# Train full three-modal QRSAN five times on all datasets, then create Figure 4.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV=""
GPUS=(0 1)
TASKS_PER_GPU=1
RUNS=5
BASE_SEED=2024
EPOCHS=""
NO_RESUME=0
CONTINUE_ON_ERROR=1
DRY_RUN=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/train_figure4_qrsan_linux.sh [options]

Default workload: full three-modal QRSAN, 3 datasets × 5 seeds = 15 runs.
Each run saves test outputs/targets, then class-level Figure 4 results are
generated automatically for CMU-MOSI, CMU-MOSEI and IEMOCAP.

Options:
  --gpus LIST              Comma-separated GPU IDs (default: 0,1)
  --tasks-per-gpu N        Concurrent workers per GPU (default: 1)
  --runs N                 Number of independent seeds (default: 5)
  --base-seed N            First seed (default: 2024)
  --epochs N               Override epoch count
  --python PATH            Python executable (default: python3)
  --conda-env NAME         Use conda run -n NAME
  --fresh                  Retrain even when metrics.csv already exists
  --check                  Run one forward/backward batch per configuration
  --stop-on-error          Stop a worker at its first failure
  --dry-run                Print jobs without training
  -h, --help               Show this help
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

while (($# > 0)); do
  case "$1" in
    --gpus) IFS=',' read -r -a GPUS <<<"$2"; shift 2 ;;
    --tasks-per-gpu) TASKS_PER_GPU="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --base-seed) BASE_SEED="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
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

[[ "${TASKS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || die "--tasks-per-gpu must be positive"
[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]] || die "--runs must be positive"
[[ "${BASE_SEED}" =~ ^[0-9]+$ ]] || die "--base-seed must be non-negative"
[[ -z "${EPOCHS}" || "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be positive"
((${#GPUS[@]} > 0)) || die "--gpus must not be empty"
for gpu_id in "${GPUS[@]}"; do [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${gpu_id}"; done

if [[ -n "${CONDA_ENV}" ]]; then
  command -v conda >/dev/null 2>&1 || die "conda was not found in PATH"
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}")
else
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
  RUNNER=("${PYTHON_BIN}")
fi

WORKER_GPUS=()
for gpu_id in "${GPUS[@]}"; do
  for ((slot=0; slot<TASKS_PER_GPU; slot++)); do WORKER_GPUS+=("${gpu_id}"); done
done

run_worker() {
  local worker_index="$1"
  local gpu_id="${WORKER_GPUS[worker_index]}"
  local command=("${RUNNER[@]}" -u scripts/train_figure4_qrsan.py
    --runs "${RUNS}" --base-seed "${BASE_SEED}"
    --worker-index "${worker_index}" --worker-count "${#WORKER_GPUS[@]}")
  [[ -z "${EPOCHS}" ]] || command+=(--epochs "${EPOCHS}")
  ((NO_RESUME == 0)) || command+=(--no-resume)
  ((CONTINUE_ON_ERROR == 0)) || command+=(--continue-on-error)
  ((CHECK_ONLY == 0)) || command+=(--check)
  ((DRY_RUN == 0)) || command+=(--dry-run)
  printf '[worker %s] GPU=%s: ' "${worker_index}" "${gpu_id}"
  printf '%q ' env "CUDA_VISIBLE_DEVICES=${gpu_id}" "${command[@]}"
  printf '\n'
  ((DRY_RUN == 1)) && return 0
  mkdir -p eval/figure4_predictions
  env CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}" 2>&1 | tee "eval/figure4_predictions/worker_${worker_index}.log"
}

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
pids=()
for ((worker=0; worker<${#WORKER_GPUS[@]}; worker++)); do
  run_worker "${worker}" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
((status == 0)) || exit "${status}"
((DRY_RUN == 0 && CHECK_ONLY == 0)) || exit 0

for dataset in cmumosi cmumosei iemocap; do
  inputs=()
  for ((run_id=1; run_id<=RUNS; run_id++)); do
    file="eval/figure4_predictions/${dataset}/t_v_a/full_qrsan/run_${run_id}/predictions.npz"
    [[ -f "${file}" ]] || die "missing prediction file: ${file}"
    inputs+=("${file}")
  done
  "${RUNNER[@]}" scripts/evaluate_figure4.py --dataset "${dataset}" \
    --inputs "${inputs[@]}" --output-dir "eval/figure4/${dataset}/qrsan"
done
