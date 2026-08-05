#!/usr/bin/env bash
# Batch-train 12 multimodal models on CMU-MOSI, CMU-MOSEI and IEMOCAP.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV=""
BATCH_COUNT=5
GPUS=(0 1)
BASE_SEED=77
EPOCHS=""
CHECK_ONLY=0
NO_RESUME=0
CONTINUE_ON_ERROR=1
DRY_RUN=0

DATASETS=(cmumosi cmumosei iemocap)
MODELS=(ef_lstm lf_lstm marn mfn tfn lmf mult qmf qrsan megakan m3sa almt)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_all_linux.sh [options]

By default, five complete batches are trained. Each batch contains all 12 models
on CMU-MOSI, CMU-MOSEI and IEMOCAP (36 runs per batch). GPU 0 and GPU 1 each
run one batch at a time. Completed CSVs are skipped independently per batch.

Options:
  --batches N              Number of complete batches (default: 5)
  --gpus LIST              Comma-separated physical GPU IDs (default: 0,1)
  --gpu ID                 Backward-compatible shorthand for --gpus ID
  --base-seed N            Seed for batch 1; increments per batch (default: 77)
  --epochs N               Override epochs in every generated config
  --datasets LIST          Comma-separated subset: cmumosi,cmumosei,iemocap
  --models LIST            Comma-separated subset, for example: mult,qmf,almt
  --python PATH            Python executable (default: python3 or $PYTHON_BIN)
  --conda-env NAME         Run through "conda run -n NAME"
  --check                  Run one forward/backward batch only; do not train
  --fresh                  Re-run completed experiments instead of resuming
  --stop-on-error          Stop when the first experiment fails
  --dry-run                Print the resolved command without executing it
  -h, --help               Show this help

Examples:
  bash scripts/train_all_linux.sh
  bash scripts/train_all_linux.sh --batches 5 --gpus 0,1 --epochs 20
  bash scripts/train_all_linux.sh --batches 1 --gpu 0
  bash scripts/train_all_linux.sh --datasets cmumosi,cmumosei --models mult,qmf,almt
  bash scripts/train_all_linux.sh --conda-env multimodal-sa --check

Outputs:
  eval/matrix_<batch_id>/<dataset>/<model>.csv  Test metrics
  eval/matrix_<batch_id>/<dataset>/<model>.log  Complete training log
  eval/matrix_<batch_id>/training_state.json    Resume/progress state
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

split_csv() {
  local value="$1"
  local -n destination="$2"
  IFS=',' read -r -a destination <<<"${value}"
  ((${#destination[@]} > 0)) || die "empty list supplied"
}

while (($# > 0)); do
  case "$1" in
    --batches)
      (($# >= 2)) || die "--batches requires a positive integer"
      [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "invalid batch count: $2"
      BATCH_COUNT="$2"
      shift 2
      ;;
    --gpus)
      (($# >= 2)) || die "--gpus requires a comma-separated list"
      split_csv "$2" GPUS
      shift 2
      ;;
    --gpu)
      (($# >= 2)) || die "--gpu requires an ID"
      GPUS=("$2")
      shift 2
      ;;
    --base-seed)
      (($# >= 2)) || die "--base-seed requires a non-negative integer"
      [[ "$2" =~ ^[0-9]+$ ]] || die "invalid base seed: $2"
      BASE_SEED="$2"
      shift 2
      ;;
    --epochs)
      (($# >= 2)) || die "--epochs requires a positive integer"
      [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "invalid epoch count: $2"
      EPOCHS="$2"
      shift 2
      ;;
    --datasets)
      (($# >= 2)) || die "--datasets requires a comma-separated list"
      split_csv "$2" DATASETS
      shift 2
      ;;
    --models)
      (($# >= 2)) || die "--models requires a comma-separated list"
      split_csv "$2" MODELS
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires an executable path"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --conda-env)
      (($# >= 2)) || die "--conda-env requires an environment name"
      CONDA_ENV="$2"
      shift 2
      ;;
    --check)
      CHECK_ONLY=1
      shift
      ;;
    --fresh)
      NO_RESUME=1
      shift
      ;;
    --stop-on-error)
      CONTINUE_ON_ERROR=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

((${#GPUS[@]} > 0)) || die "at least one GPU ID is required"
for gpu_id in "${GPUS[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${gpu_id}"
done

[[ -f "${PROJECT_ROOT}/scripts/train_dataset_matrix.py" ]] || \
  die "missing scripts/train_dataset_matrix.py under ${PROJECT_ROOT}"
[[ -f "${PROJECT_ROOT}/run.py" ]] || die "missing run.py under ${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1

if [[ -n "${CONDA_ENV}" ]]; then
  command -v conda >/dev/null 2>&1 || die "conda was not found in PATH"
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}")
else
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
  RUNNER=("${PYTHON_BIN}")
fi

printf 'Project:  %s\n' "${PROJECT_ROOT}"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Models:   %s\n' "${MODELS[*]}"
printf 'Batches:  %d\n' "${BATCH_COUNT}"
printf 'GPUs:     %s\n' "${GPUS[*]}"
printf 'Runs:     %d\n' "$(( BATCH_COUNT * ${#DATASETS[@]} * ${#MODELS[@]} ))"

on_interrupt() {
  printf '\nInterrupted. Progress remains in eval/matrix_<batch_id>/.\n' >&2
  jobs -pr | xargs -r kill 2>/dev/null || true
  exit 130
}
trap on_interrupt INT TERM

cd "${PROJECT_ROOT}"

build_command() {
  local batch_id="$1"
  local seed="$2"
  BATCH_COMMAND=(
    "${RUNNER[@]}"
    -u scripts/train_dataset_matrix.py
    --batch-id "${batch_id}"
    --seed "${seed}"
    --datasets "${DATASETS[@]}"
    --models "${MODELS[@]}"
  )
  [[ -z "${EPOCHS}" ]] || BATCH_COMMAND+=(--epochs "${EPOCHS}")
  ((CHECK_ONLY == 0)) || BATCH_COMMAND+=(--check)
  ((NO_RESUME == 0)) || BATCH_COMMAND+=(--no-resume)
  ((CONTINUE_ON_ERROR == 0)) || BATCH_COMMAND+=(--continue-on-error)
}

run_batch() {
  local batch_id="$1"
  local gpu_id="$2"
  local seed=$((BASE_SEED + batch_id - 1))
  local result_dir="${PROJECT_ROOT}/eval/matrix_${batch_id}"
  local wrapper_log="${result_dir}/batch_$(date '+%Y%m%d_%H%M%S').log"
  build_command "${batch_id}" "${seed}"
  printf '[batch %d] GPU=%s seed=%d output=%s\n' \
    "${batch_id}" "${gpu_id}" "${seed}" "${result_dir}"
  printf '[batch %d] Command: ' "${batch_id}"
  printf '%q ' env "CUDA_VISIBLE_DEVICES=${gpu_id}" "${BATCH_COMMAND[@]}"
  printf '\n'
  ((DRY_RUN)) && return 0
  mkdir -p "${result_dir}"
  env CUDA_VISIBLE_DEVICES="${gpu_id}" "${BATCH_COMMAND[@]}" \
    2>&1 | tee "${wrapper_log}"
}

run_worker() {
  local slot="$1"
  local gpu_id="${GPUS[slot]}"
  local batch_id
  local worker_status=0
  for ((batch_id=slot + 1; batch_id<=BATCH_COUNT; batch_id+=${#GPUS[@]})); do
    if ! run_batch "${batch_id}" "${gpu_id}"; then
      worker_status=1
      printf '[batch %d] FAILED on GPU %s\n' "${batch_id}" "${gpu_id}" >&2
      ((CONTINUE_ON_ERROR == 1)) || break
    fi
  done
  return "${worker_status}"
}

if ((DRY_RUN)); then
  for ((slot=0; slot<${#GPUS[@]} && slot<BATCH_COUNT; slot++)); do
    run_worker "${slot}"
  done
  exit 0
fi

WORKER_PIDS=()
for ((slot=0; slot<${#GPUS[@]} && slot<BATCH_COUNT; slot++)); do
  run_worker "${slot}" &
  WORKER_PIDS+=("$!")
done

FINAL_STATUS=0
for worker_pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${worker_pid}"; then
    FINAL_STATUS=1
  fi
done

if ((FINAL_STATUS == 0)); then
  printf 'All %d batches completed successfully.\n' "${BATCH_COUNT}"
else
  printf 'One or more batches failed. Inspect eval/matrix_<batch_id>/*.log.\n' >&2
fi
exit "${FINAL_STATUS}"
