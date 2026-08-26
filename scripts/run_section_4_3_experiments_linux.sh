#!/usr/bin/env bash
# One Linux entry point for every Section 4.3 robustness experiment.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV=""
RUNS=5
BASE_SEED=77
EPOCHS=""
INPUT_ROOT="eval"
OUTPUT_ROOT="eval/section_4_3"
TABLE_LABEL="Sx"
BOOTSTRAP_RESAMPLES=10000
PERMUTATION_RESAMPLES=10000
STATISTICAL_RANDOM_SEED=20260818
UNIFORM_RANDOM_DRAWS=1000
UNIFORM_RANDOM_SEED=20260818
ANALYZE_ONLY=0
FRESH=0
GPUS=(0 1)
TASKS_PER_GPU=2
METRICS=(accuracy balanced_accuracy macro_f1)
COMPARATORS=(cmumosei=almt cmumosi=megakan iemocap=ef_lstm)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_section_4_3_experiments_linux.sh [options]

Run the complete Section 4.3 experiment once. The default workflow trains the
six required model/dataset pairs for five matched seeds (30 runs total), then:
  1. creates majority-class and uniform-random baseline tables for CMU-MOSEI,
     CMU-MOSI, and IEMOCAP;
  2. creates paired bootstrap/permutation comparisons and Holm correction
     across QRSAN and the designated comparator on all three datasets.

Options:
  --analyze-only           Skip training and analyse existing saved predictions
  --fresh                  Re-run completed experiments rather than resuming them
  --gpus LIST              Comma-separated physical GPU IDs for seed-level parallelism
                           (default: 0,1; use --gpus 0 for one GPU)
  --tasks-per-gpu N        Concurrent seed batches per GPU (default: 2)
  --runs N                 Number of matched seeds (default: 5)
  --base-seed N            Seed for run 1; increments per run (default: 77)
  --epochs N               Override the configured epoch count during training
  --input-root PATH        Root containing matrix_<id>/... predictions (default: eval)
  --output-root PATH       Root for both Section 4.3 reports (default: eval/section_4_3)
  --table-label LABEL      Table reference used in generated prose (default: Sx)
  --metrics LIST           Comma-separated metrics (default: accuracy,balanced_accuracy,macro_f1)
  --comparators LIST       Comma-separated DATASET=MODEL mappings
                          (default: cmumosei=almt,cmumosi=megakan,iemocap=ef_lstm)
  --bootstrap-resamples N  Paired bootstrap draws, at least 10000 (default: 10000)
  --permutation-resamples N Paired permutation draws, at least 10000 (default: 10000)
  --statistical-random-seed N Resampling seed for paired statistics (default: 20260818)
  --uniform-random-draws N Monte Carlo draws for each random baseline (default: 1000)
  --uniform-random-seed N  Seed for all random baselines (default: 20260818)
  --python PATH            Python executable (default: python3 or $PYTHON_BIN)
  --conda-env NAME         Execute through "conda run -n NAME"
  -h, --help               Show this help

Examples:
  # Full experiment: 30 matched model/dataset runs, then both analyses.
  bash scripts/run_section_4_3_experiments_linux.sh \
    --gpus 0,1 --tasks-per-gpu 2 --conda-env multimodal-sa --table-label S4

  # Re-use existing five-seed prediction files only.
  bash scripts/run_section_4_3_experiments_linux.sh \
    --analyze-only --conda-env multimodal-sa --table-label S4
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

split_csv() {
  local value="$1"
  local destination="$2"
  IFS=',' read -r -a "${destination}" <<<"${value}"
  local -n values="${destination}"
  ((${#values[@]} > 0)) || die "empty comma-separated list"
}

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

at_least_10000() {
  nonnegative_integer "$1" && ((10#$1 >= 10000))
}

at_least_two() {
  nonnegative_integer "$1" && ((10#$1 >= 2))
}

relative_project_path() {
  [[ "$1" != /* && "$1" != *".."* ]]
}

while (($# > 0)); do
  case "$1" in
    --analyze-only)
      ANALYZE_ONLY=1
      shift
      ;;
    --fresh)
      FRESH=1
      shift
      ;;
    --gpus)
      (($# >= 2)) || die "--gpus requires a comma-separated list"
      split_csv "$2" GPUS
      shift 2
      ;;
    --tasks-per-gpu)
      (($# >= 2)) || die "--tasks-per-gpu requires a positive integer"
      positive_integer "$2" || die "invalid --tasks-per-gpu value: $2"
      TASKS_PER_GPU="$2"
      shift 2
      ;;
    --runs)
      (($# >= 2)) || die "--runs requires a positive integer"
      positive_integer "$2" || die "invalid --runs value: $2"
      RUNS="$2"
      shift 2
      ;;
    --base-seed)
      (($# >= 2)) || die "--base-seed requires a non-negative integer"
      nonnegative_integer "$2" || die "invalid --base-seed value: $2"
      BASE_SEED="$2"
      shift 2
      ;;
    --epochs)
      (($# >= 2)) || die "--epochs requires a positive integer"
      positive_integer "$2" || die "invalid --epochs value: $2"
      EPOCHS="$2"
      shift 2
      ;;
    --input-root)
      (($# >= 2)) || die "--input-root requires a relative project path"
      relative_project_path "$2" || die "--input-root must remain within the project"
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a relative project path"
      relative_project_path "$2" || die "--output-root must remain within the project"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --table-label)
      (($# >= 2)) || die "--table-label requires a value"
      [[ "$2" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid --table-label value: $2"
      TABLE_LABEL="$2"
      shift 2
      ;;
    --metrics)
      (($# >= 2)) || die "--metrics requires a comma-separated list"
      split_csv "$2" METRICS
      shift 2
      ;;
    --comparators)
      (($# >= 2)) || die "--comparators requires a comma-separated list"
      split_csv "$2" COMPARATORS
      shift 2
      ;;
    --bootstrap-resamples)
      (($# >= 2)) || die "--bootstrap-resamples requires an integer >= 10000"
      at_least_10000 "$2" || die "invalid bootstrap resample count: $2"
      BOOTSTRAP_RESAMPLES="$2"
      shift 2
      ;;
    --permutation-resamples)
      (($# >= 2)) || die "--permutation-resamples requires an integer >= 10000"
      at_least_10000 "$2" || die "invalid permutation resample count: $2"
      PERMUTATION_RESAMPLES="$2"
      shift 2
      ;;
    --statistical-random-seed)
      (($# >= 2)) || die "--statistical-random-seed requires a non-negative integer"
      nonnegative_integer "$2" || die "invalid random seed: $2"
      STATISTICAL_RANDOM_SEED="$2"
      shift 2
      ;;
    --uniform-random-draws)
      (($# >= 2)) || die "--uniform-random-draws requires an integer >= 2"
      at_least_two "$2" || die "invalid draw count: $2"
      UNIFORM_RANDOM_DRAWS="$2"
      shift 2
      ;;
    --uniform-random-seed)
      (($# >= 2)) || die "--uniform-random-seed requires a non-negative integer"
      nonnegative_integer "$2" || die "invalid random seed: $2"
      UNIFORM_RANDOM_SEED="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a path or command"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --conda-env)
      (($# >= 2)) || die "--conda-env requires an environment name"
      CONDA_ENV="$2"
      shift 2
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

if ((ANALYZE_ONLY)) && ((FRESH)); then
  die "--analyze-only and --fresh cannot be used together"
fi
if ((ANALYZE_ONLY == 0)) && [[ "${INPUT_ROOT}" != "eval" ]]; then
  die "training writes to eval, so --input-root must remain eval unless --analyze-only is used"
fi
if [[ -n "${CONDA_ENV}" ]]; then
  command -v conda >/dev/null 2>&1 || die "conda was not found in PATH"
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}")
elif [[ -x "${PYTHON_BIN}" ]]; then
  RUNNER=("${PYTHON_BIN}")
else
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable was not found: ${PYTHON_BIN}"
  RUNNER=("${PYTHON_BIN}")
fi

for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU ID: ${gpu}"
done

[[ -f "${PROJECT_ROOT}/scripts/train_dataset_matrix.py" ]] || die "training script is missing"
[[ -f "${PROJECT_ROOT}/scripts/report_class_imbalance_robustness.py" ]] || die "class-imbalance report script is missing"
[[ -f "${PROJECT_ROOT}/scripts/paired_prediction_statistics.py" ]] || die "statistical script is missing"
for dataset in cmumosei cmumosi iemocap; do
  [[ -f "${PROJECT_ROOT}/data/cmumosi_cmumosei_iemocap_mult/${dataset}_data.pkl" ]] || \
    die "${dataset} data pickle is missing"
done
cd "${PROJECT_ROOT}"

declare -A COMPARATOR_BY_DATASET=()
for mapping in "${COMPARATORS[@]}"; do
  [[ "${mapping}" == *=* ]] || die "invalid comparator mapping: ${mapping}"
  dataset="${mapping%%=*}"
  model="${mapping#*=}"
  [[ "${dataset}" =~ ^(cmumosei|cmumosi|iemocap)$ && "${model}" =~ ^[A-Za-z0-9_.-]+$ ]] || \
    die "invalid comparator mapping: ${mapping}"
  [[ -z "${COMPARATOR_BY_DATASET[${dataset}]:-}" ]] || die "duplicate comparator for ${dataset}"
  COMPARATOR_BY_DATASET["${dataset}"]="${model}"
done
for dataset in cmumosei cmumosi iemocap; do
  [[ -n "${COMPARATOR_BY_DATASET[${dataset}]:-}" ]] || die "missing comparator for ${dataset}"
done

train_seed_batch() {
  local batch_id="$1"
  local gpu="$2"
  local seed=$((10#${BASE_SEED} + batch_id - 1))
  local dataset comparator

  printf '[batch %d/%d] GPU %s, seed %d\n' \
    "${batch_id}" "${RUNS}" "${gpu}" "${seed}"
  for dataset in cmumosei cmumosi iemocap; do
    comparator="${COMPARATOR_BY_DATASET[${dataset}]}"
    local training_command=(
      "${RUNNER[@]}"
      -u scripts/train_dataset_matrix.py
      --batch-id "${batch_id}"
      --seed "${seed}"
      --datasets "${dataset}"
      --models qrsan "${comparator}"
    )
    if ((FRESH)); then
      training_command+=(--no-resume)
    fi
    [[ -z "${EPOCHS}" ]] || training_command+=(--epochs "${EPOCHS}")
    CUDA_VISIBLE_DEVICES="${gpu}" "${training_command[@]}"
  done
}

run_gpu_worker() {
  local worker_index="$1"
  local gpu="$2"
  local batch_id

  for ((batch_id = worker_index + 1; batch_id <= RUNS; batch_id += worker_slots)); do
    train_seed_batch "${batch_id}" "${gpu}"
  done
}

if ((ANALYZE_ONLY == 0)); then
  worker_slots=$((${#GPUS[@]} * TASKS_PER_GPU))
  printf 'Training %d matched seed batches across GPU(s): %s (%d concurrent batch(es) per GPU)\n' \
    "${RUNS}" "${GPUS[*]}" "${TASKS_PER_GPU}"
  worker_pids=()
  for ((worker_index = 0; worker_index < worker_slots && worker_index < RUNS; worker_index++)); do
    gpu_index=$((worker_index % ${#GPUS[@]}))
    run_gpu_worker "${worker_index}" "${GPUS[gpu_index]}" &
    worker_pids+=("$!")
  done

  training_failed=0
  for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
      training_failed=1
    fi
  done
  ((training_failed == 0)) || die "one or more GPU workers failed; inspect eval/matrix_*/<dataset>/*.log"
fi

batch_ids=()
for ((batch_id = 1; batch_id <= RUNS; batch_id++)); do
  batch_ids+=("${batch_id}")
done

for dataset in cmumosei cmumosi iemocap; do
  qrsan_predictions=()
  for ((batch_id = 1; batch_id <= RUNS; batch_id++)); do
    qrsan_predictions+=("${INPUT_ROOT}/matrix_${batch_id}/${dataset}/qrsan.predictions.npz")
  done
  robustness_command=(
    "${RUNNER[@]}"
    scripts/report_class_imbalance_robustness.py
    --dataset "${dataset}"
    --dataset-pickle "data/cmumosi_cmumosei_iemocap_mult/${dataset}_data.pkl"
    --prediction-files "${qrsan_predictions[@]}"
    --model-name QRSAN
    --uniform-random-draws "${UNIFORM_RANDOM_DRAWS}"
    --random-seed "${UNIFORM_RANDOM_SEED}"
    --output-dir "${OUTPUT_ROOT}/${dataset}_class_imbalance_robustness"
  )
  "${robustness_command[@]}"
done

statistics_command=(
  "${RUNNER[@]}"
  scripts/paired_prediction_statistics.py
  --input-root "${INPUT_ROOT}"
  --output-dir "${OUTPUT_ROOT}/paired_statistics"
  --batch-ids "${batch_ids[@]}"
  --datasets cmumosei cmumosi iemocap
  --comparators "${COMPARATORS[@]}"
  --metrics "${METRICS[@]}"
  --bootstrap-resamples "${BOOTSTRAP_RESAMPLES}"
  --permutation-resamples "${PERMUTATION_RESAMPLES}"
  --random-seed "${STATISTICAL_RANDOM_SEED}"
  --table-label "${TABLE_LABEL}"
)
"${statistics_command[@]}"

printf 'Completed. Read these generated files:\n'
for dataset in cmumosei cmumosi iemocap; do
  printf '  %s/%s_class_imbalance_robustness/section_4_3_table.md\n' "${OUTPUT_ROOT}" "${dataset}"
done
printf '  %s/paired_statistics/paired_statistical_comparisons.md\n' "${OUTPUT_ROOT}"
printf '  %s/paired_statistics/section_4_3_replacement.md\n' "${OUTPUT_ROOT}"
printf '  %s/paired_statistics/reviewer_response_snippet.md\n' "${OUTPUT_ROOT}"
