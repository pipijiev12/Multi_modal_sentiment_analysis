#!/usr/bin/env bash
# Re-run only the seven projection-mapping jobs whose artifacts were created
# by an overlapping/failed process and therefore have return_code != 0.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV="multimodal-sa"
GPUS="0,1,2,3"

usage() {
  cat <<'EOF'
Usage: bash experiment1/rerun_projection_mapping_repairs_linux.sh [--conda-env ENV] [--gpus 0,1,2,3]

Re-runs exactly seven projection_mapping jobs with --no-resume.  It does not
touch any other Experiment 1 result.  Logs are written to
eval/experiment1/projection_mapping/repair_*.log.
EOF
}

while (($#)); do
  case "$1" in
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
((${#GPU_LIST[@]} >= 1)) || { echo "ERROR: provide at least one GPU ID" >&2; exit 2; }

if ! grep -q 'Checkpoint exists without a best model' utils/model.py; then
  echo "ERROR: utils/model.py lacks the incomplete-checkpoint recovery fix; sync commit 5d57c34 first." >&2
  exit 2
fi
if pgrep -af 'run.py -config .*experiment1/generated' >/dev/null; then
  echo "ERROR: an Experiment 1 run.py process is already active. Wait for it to finish before this targeted repair." >&2
  pgrep -af 'run.py -config .*experiment1/generated' >&2
  exit 2
fi

RUNNER=(conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN")
LOG_DIR="eval/experiment1/projection_mapping"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1" seed="$2" dataset="$3" model="$4"
  local log_file="$LOG_DIR/repair_seed${seed}_${dataset}_${model}.log"
  printf 'RUN GPU=%s seed=%s dataset=%s model=%s\n' "$gpu" "$seed" "$dataset" "$model" | tee "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "${RUNNER[@]}" experiment1/run_experiment1.py \
    --suite projection_mapping --seeds "$seed" --datasets "$dataset" --models "$model" --no-resume \
    >> "$log_file" 2>&1
}

wait_batch() {
  local failed=0 pid
  for pid in "$@"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

# Each row is: seed dataset model.  Run the four slowest candidates first.
jobs=(
  '77 cmumosei born_score_mlp'
  '77 cmumosei learned_complex_score_mlp'
  '78 cmumosei learned_complex_score_mlp'
  '78 iemocap born_score_mlp'
  '79 cmumosei born_score_mlp'
  '80 cmumosei born_score_mlp'
  '80 cmumosei learned_complex_score_mlp'
)

overall_failed=0
for ((offset=0; offset<${#jobs[@]}; offset+=${#GPU_LIST[@]})); do
  pids=()
  for ((slot=0; slot<${#GPU_LIST[@]} && offset+slot<${#jobs[@]}; slot++)); do
    read -r seed dataset model <<< "${jobs[$((offset+slot))]}"
    run_one "${GPU_LIST[$slot]}" "$seed" "$dataset" "$model" &
    pids+=("$!")
  done
  wait_batch "${pids[@]}" || overall_failed=1
done

if ((overall_failed)); then
  echo "One or more repair jobs failed; inspect $LOG_DIR/repair_*.log" >&2
  exit 1
fi
echo "All seven projection-mapping repair jobs completed. Run validate_experiment1_results.py --require-all next."
