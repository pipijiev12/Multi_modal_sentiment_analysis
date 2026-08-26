#!/usr/bin/env bash
# Single Linux entry point for all currently executable reviewer experiments.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV=""
GPUS="0,1"
TASKS_PER_GPU=1
EPOCHS=""
TABLE_LABEL="S4"
IEMOCAP_METADATA=""
ANALYZE_ONLY=0
FRESH=0
SKIP_SECTION43=0
SKIP_EXPERIMENT1=0

usage() {
  cat <<'EOF'
Usage: bash experiment1/run_all_reviewer_experiments_linux.sh [options]

Runs Section 4.3 first, then Experiment 1. Both stages use the supplied GPUs
internally but never run concurrently, preventing GPU oversubscription.

Options:
  --gpus LIST                 Physical GPU IDs, e.g. 0,1 (default: 0,1)
  --tasks-per-gpu N           Concurrent seed batches per GPU (default: 1)
  --conda-env NAME            Conda environment name
  --epochs N                  Training epoch override for both stages
  --table-label LABEL         Section 4.3 table label (default: S4)
  --iemocap-metadata FILE     CSV with utterance_id,session,speaker; generate LOSO manifests
  --analyze-only              Do not train; regenerate analyses from saved artifacts
  --fresh                     Re-run completed model experiments and overwrite their artifacts
  --skip-section43            Skip the Section 4.3 stage
  --skip-experiment1          Skip the Experiment 1 stage
  -h, --help                  Show this message

Note: --iemocap-metadata generates speaker-disjoint fold manifests only. It
does not train LOSO folds until the data reader is configured to consume them.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (($#)); do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --tasks-per-gpu) TASKS_PER_GPU="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --table-label) TABLE_LABEL="$2"; shift 2 ;;
    --iemocap-metadata) IEMOCAP_METADATA="$2"; shift 2 ;;
    --analyze-only) ANALYZE_ONLY=1; shift ;;
    --fresh) FRESH=1; shift ;;
    --skip-section43) SKIP_SECTION43=1; shift ;;
    --skip-experiment1) SKIP_EXPERIMENT1=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$TASKS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "--tasks-per-gpu must be a positive integer"
[[ -z "$IEMOCAP_METADATA" || -f "$IEMOCAP_METADATA" ]] || die "IEMOCAP metadata CSV not found: $IEMOCAP_METADATA"
cd "$ROOT"

common=(--gpus "$GPUS" --tasks-per-gpu "$TASKS_PER_GPU")
[[ -z "$CONDA_ENV" ]] || common+=(--conda-env "$CONDA_ENV")
[[ -z "$EPOCHS" ]] || common+=(--epochs "$EPOCHS")
[[ $ANALYZE_ONLY -eq 0 ]] || common+=(--analyze-only)
((FRESH == 0)) || common+=(--fresh)

if (( ! SKIP_SECTION43 )); then
  section_command=(bash scripts/run_section_4_3_experiments_linux.sh "${common[@]}" --table-label "$TABLE_LABEL")
  printf '== Stage 1/2: Section 4.3 ==\n'
  "${section_command[@]}"
fi

if (( ! SKIP_EXPERIMENT1 )); then
  experiment_command=(bash experiment1/run_experiment1_linux.sh "${common[@]}")
  printf '== Stage 2/2: Experiment 1 ==\n'
  "${experiment_command[@]}"
fi

if [[ -n "$IEMOCAP_METADATA" ]]; then
  if [[ -n "$CONDA_ENV" ]]; then runner=(conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN"); else runner=("$PYTHON_BIN"); fi
  "${runner[@]}" experiment1/make_iemocap_loso_folds.py --metadata-csv "$IEMOCAP_METADATA" --output-dir eval/experiment1/iemocap_loso_folds
fi

printf 'Completed reviewer-experiment orchestration.\n'
