#!/usr/bin/env bash
# Batch entry point for Experiment 1.  Training is scheduled at the individual
# seed/dataset/model level so --tasks-per-gpu is a real per-GPU concurrency cap.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"; CONDA_ENV=""; GPUS="0,1"; TASKS=1; ANALYZE_ONLY=0; EPOCHS=""; FRESH=0; OUTPUT_ROOT="eval/experiment1"
usage(){ cat <<'EOF'
Usage: bash experiment1/run_experiment1_linux.sh [--conda-env ENV] [--gpus 0,1] [--tasks-per-gpu N] [--epochs N] [--output-root DIR] [--fresh] [--analyze-only]
EOF
}
while (($#)); do case "$1" in --conda-env) CONDA_ENV="$2"; shift 2;; --gpus) GPUS="$2"; shift 2;; --tasks-per-gpu) TASKS="$2"; shift 2;; --epochs) EPOCHS="$2"; shift 2;; --output-root) OUTPUT_ROOT="$2"; shift 2;; --fresh) FRESH=1; shift;; --analyze-only) ANALYZE_ONLY=1; shift;; -h|--help) usage; exit 0;; *) echo "Unknown option: $1" >&2; exit 2;; esac; done
if [[ -n "$CONDA_ENV" ]]; then RUNNER=(conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN"); else RUNNER=("$PYTHON_BIN"); fi
cd "$ROOT"; IFS=',' read -r -a GPU_LIST <<< "$GPUS"
[[ "$OUTPUT_ROOT" != /* && "$OUTPUT_ROOT" != *".."* ]] || { echo "ERROR: --output-root must be a relative repository path" >&2; exit 2; }
run_job(){
  local suite="$1" seed="$2" dataset="$3" model="$4" gpu="$5"
  local command=("${RUNNER[@]}" experiment1/run_experiment1.py --suite "$suite" --seeds "$seed" --datasets "$dataset" --models "$model" --output-root "$OUTPUT_ROOT")
  [[ -z "$EPOCHS" ]] || command+=(--epochs "$EPOCHS")
  ((FRESH == 0)) || command+=(--no-resume)
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"
}
if (( ! ANALYZE_ONLY )); then
  suites=(main ablation_available projection_head projection_mapping efficiency)
  worker_slots=$((${#GPU_LIST[@]} * TASKS))
  queue_dir="$(mktemp -d "${TMPDIR:-/tmp}/qrsan-experiment1.XXXXXX")"
  # Background workers inherit shell traps.  Restrict cleanup to this parent
  # shell; otherwise a worker exiting would delete the shared queue mid-run.
  queue_owner_bashpid="$BASHPID"
  cleanup_queue(){ [[ "$BASHPID" != "$queue_owner_bashpid" ]] || rm -rf -- "$queue_dir"; }
  trap cleanup_queue EXIT
  job_number=0
  for suite in "${suites[@]}"; do
    while IFS=$'\t' read -r seed dataset model; do
      printf '%s\t%s\t%s\t%s\n' "$suite" "$seed" "$dataset" "$model" > "$queue_dir/job_$(printf '%06d' "$job_number")"
      ((job_number+=1))
    done < <("${RUNNER[@]}" experiment1/run_experiment1.py --suite "$suite" --list-jobs --output-root "$OUTPUT_ROOT")
  done
  printf 'Queued %s Experiment 1 jobs across %s worker(s) (%s task(s) per GPU).\n' "$job_number" "$worker_slots" "$TASKS"
  run_worker(){
    local gpu="$1" job_file running_file suite seed dataset model
    worker_error(){
      local status="$?"
      printf 'ERROR: scheduler worker on GPU %s exited at line %s (status %s): %s\n' \
        "$gpu" "$LINENO" "$status" "$BASH_COMMAND" >&2
    }
    trap worker_error ERR
    while :; do
      exec 9>"$queue_dir/.lock"
      flock -x 9
      # Only queued files are exactly job_000000 through job_999999.  A
      # claimed job is renamed to *.running.<pid> and must never be claimed by
      # another worker.
      # Do not use head here: with `set -o pipefail`, head closes the pipe
      # early and makes sort/find exit with SIGPIPE (status 141).
      job_file="$(find "$queue_dir" -maxdepth 1 -type f -name 'job_[0-9][0-9][0-9][0-9][0-9][0-9]' -print | sort | sed -n '1p')"
      if [[ -n "$job_file" ]]; then
        running_file="${job_file}.running.$$"
        mv -- "$job_file" "$running_file"
      fi
      flock -u 9
      exec 9>&-
      [[ -n "$job_file" ]] || break
      IFS=$'\t' read -r suite seed dataset model < "$running_file"
      rm -f -- "$running_file"
      if ! run_job "$suite" "$seed" "$dataset" "$model" "$gpu"; then
        printf '%s\t%s\t%s\t%s\tGPU%s\n' "$suite" "$seed" "$dataset" "$model" "$gpu" >> "$queue_dir/failed_jobs.tsv"
        printf 'ERROR: job failed: %s seed=%s %s/%s on GPU %s; continuing queue.\n' "$suite" "$seed" "$dataset" "$model" "$gpu" >&2
      fi
    done
  }
  pids=(); for ((worker=0; worker<worker_slots; worker++)); do
    run_worker "${GPU_LIST[$((worker % ${#GPU_LIST[@]}))]}" & pids+=("$!")
  done
  failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [[ -s "$queue_dir/failed_jobs.tsv" ]]; then
    cp "$queue_dir/failed_jobs.tsv" "$OUTPUT_ROOT/failed_jobs.tsv"
    echo "Experiment training completed with failed jobs; inspect $OUTPUT_ROOT/failed_jobs.tsv and the matching run.log files." >&2
    failed=1
  fi
  ((failed==0)) || { echo "Experiment training failed; inspect $OUTPUT_ROOT/*/*/*/*/run.log" >&2; exit 1; }
  mapfile -t benchmark_configs < <(find experiment1/generated/efficiency/seed_77 -name '*.ini' -type f | sort)
  if ((${#benchmark_configs[@]})); then
    "${RUNNER[@]}" experiment1/benchmark_models.py --configs "${benchmark_configs[@]}" --output "$OUTPUT_ROOT/benchmarks/efficiency_seed77.csv"
  fi
  for dataset in cmumosei cmumosi iemocap; do
    config="experiment1/generated/main/seed_77/${dataset}/qrsan.ini"
    model_file="$OUTPUT_ROOT/main/seed_77/${dataset}/qrsan/best_model.pt"
    if [[ -f "$config" && -f "$model_file" ]]; then
      "${RUNNER[@]}" experiment1/basis_permutation_sensitivity.py --config "$config" --model-file "$model_file" --output-dir "$OUTPUT_ROOT/basis_sensitivity/${dataset}/seed_77" --permutations 10
    fi
  done
fi
"${RUNNER[@]}" experiment1/audit_predictions.py --input-root eval --output-dir "$OUTPUT_ROOT/audits/predictions"
"${RUNNER[@]}" experiment1/iemocap_split_audit.py --dataset-pickle data/cmumosi_cmumosei_iemocap_mult/iemocap_data.pkl --output-dir "$OUTPUT_ROOT/audits/iemocap"
"${RUNNER[@]}" experiment1/collect_reproducibility.py --config-root experiment1/generated --output-dir "$OUTPUT_ROOT/audits/reproducibility"
echo "Completed Experiment 1. Review eval/experiment1/audits and blocked_variants.json before manuscript claims."
