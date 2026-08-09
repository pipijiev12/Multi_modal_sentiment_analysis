#!/usr/bin/env bash
# Generate Figure 4 class-level metrics for CMU-MOSI, CMU-MOSEI and IEMOCAP.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV=""
MODEL="mult"
INPUT_ROOT="eval/mult"
OUTPUT_ROOT="eval/figure4"
BATCH_IDS=(1 2 3 4 5)

usage() {
  cat <<'EOF'
Usage: bash scripts/evaluate_figure4_linux.sh [options]

Reads saved .predictions.npz files from five matrix runs and produces the
class-level Precision, Recall, F1, confusion matrices, mean±std CSVs and PNGs
for CMU-MOSI, CMU-MOSEI and IEMOCAP.

Options:
  --model NAME            Model directory/file prefix (default: mult)
  --batch-ids LIST        Comma-separated matrix IDs (default: 1,2,3,4,5)
  --input-root PATH       Matrix result root (default: eval/mult)
  --output-root PATH      Figure 4 output root (default: eval/figure4)
  --python PATH           Python executable (default: python3)
  --conda-env NAME        Run Python via conda environment NAME
  -h, --help              Show this help
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

while (($# > 0)); do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --batch-ids) IFS=',' read -r -a BATCH_IDS <<<"$2"; shift 2 ;;
    --input-root) INPUT_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

((${#BATCH_IDS[@]} > 0)) || die "--batch-ids must not be empty"
if [[ -n "${CONDA_ENV}" ]]; then
  command -v conda >/dev/null 2>&1 || die "conda was not found in PATH"
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}")
else
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
  RUNNER=("${PYTHON_BIN}")
fi

cd "${PROJECT_ROOT}"
for dataset in cmumosi cmumosei iemocap; do
  prediction_files=()
  for batch_id in "${BATCH_IDS[@]}"; do
    prediction_file="${INPUT_ROOT}/matrix_${batch_id}/${dataset}/${MODEL}.predictions.npz"
    [[ -f "${prediction_file}" ]] || die "missing prediction file: ${prediction_file}"
    prediction_files+=("${prediction_file}")
  done
  output_dir="${OUTPUT_ROOT}/${dataset}/${MODEL}"
  printf 'Evaluating %s (%s runs) -> %s\n' "${dataset}" "${#prediction_files[@]}" "${output_dir}"
  "${RUNNER[@]}" scripts/evaluate_figure4.py \
    --dataset "${dataset}" \
    --inputs "${prediction_files[@]}" \
    --output-dir "${output_dir}"
done
