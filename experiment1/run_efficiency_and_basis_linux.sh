#!/usr/bin/env bash
# Run only the two reviewer-requested supplementary experiments:
# (1) fair efficiency training + one fixed-hardware microbenchmark per dataset;
# (2) fixed-checkpoint random interaction-basis permutation sensitivity.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV=""; GPUS="0"; TASKS=1; OUTPUT_ROOT="eval/experiment1"; FRESH=0
WARMUP=10; REPEATS=30; PERMUTATIONS=10

usage() {
  cat <<'EOF'
Usage: bash experiment1/run_efficiency_and_basis_linux.sh [options]

Options:
  --conda-env ENV       Conda environment (e.g. multimodal-sa)
  --gpus IDS            GPU ids, comma separated (default: 0)
  --tasks-per-gpu N     Concurrent efficiency-training jobs per GPU (default: 1)
  --output-root DIR     Relative output root (default: eval/experiment1)
  --fresh               Re-run completed efficiency jobs
  --warmup N            Warm-up steps for benchmark (default: 10)
  --repeats N           Timed steps for benchmark (default: 30)
  --permutations N      Random basis permutations per dataset (default: 10)
EOF
}

while (($#)); do case "$1" in
  --conda-env) CONDA_ENV="$2"; shift 2 ;;
  --gpus) GPUS="$2"; shift 2 ;;
  --tasks-per-gpu) TASKS="$2"; shift 2 ;;
  --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
  --fresh) FRESH=1; shift ;;
  --warmup) WARMUP="$2"; shift 2 ;;
  --repeats) REPEATS="$2"; shift 2 ;;
  --permutations) PERMUTATIONS="$2"; shift 2 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac; done

[[ "$OUTPUT_ROOT" != /* && "$OUTPUT_ROOT" != *".."* ]] || { echo "--output-root must be relative" >&2; exit 2; }
[[ "$TASKS" =~ ^[1-9][0-9]*$ && "$WARMUP" =~ ^[0-9]+$ && "$REPEATS" =~ ^[1-9][0-9]*$ && "$PERMUTATIONS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid numeric option" >&2; exit 2; }
cd "$ROOT"; IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ -n "$CONDA_ENV" ]]; then RUNNER=(conda run --no-capture-output -n "$CONDA_ENV" python3); else RUNNER=(python3); fi

"${RUNNER[@]}" experiment1/collect_hardware_environment.py --output-dir "$OUTPUT_ROOT/audits/hardware"

# Launch training jobs in a bounded worker pool. The runner owns per-job logs,
# checkpoints and reproducibility.json files under OUTPUT_ROOT.
mapfile -t JOBS < <("${RUNNER[@]}" experiment1/run_experiment1.py --suite efficiency --list-jobs --output-root "$OUTPUT_ROOT")
printf 'Queued %s efficiency jobs across %s worker(s).\n' "${#JOBS[@]}" "$(( ${#GPU_LIST[@]} * TASKS ))"
queue="$(mktemp -d "${TMPDIR:-/tmp}/qrsan-efficiency.XXXXXX")"; trap 'rm -rf -- "$queue"' EXIT
for i in "${!JOBS[@]}"; do printf '%s\n' "${JOBS[$i]}" > "$queue/job_$(printf '%06d' "$i")"; done
worker() {
  local gpu="$1" file line seed dataset model cmd
  while :; do
    exec 9>"$queue/.lock"; flock -x 9
    file="$(find "$queue" -maxdepth 1 -type f -name 'job_[0-9][0-9][0-9][0-9][0-9][0-9]' -print | sort | sed -n '1p')"
    [[ -z "$file" ]] || mv -- "$file" "$file.running.$$"
    flock -u 9; exec 9>&-
    [[ -n "$file" ]] || break
    file="$file.running.$$"; line="$(<"$file")"; rm -f -- "$file"
    IFS=$'\t' read -r seed dataset model <<< "$line"
    cmd=("${RUNNER[@]}" experiment1/run_experiment1.py --suite efficiency --seeds "$seed" --datasets "$dataset" --models "$model" --output-root "$OUTPUT_ROOT")
    (( FRESH == 0 )) || cmd+=(--no-resume)
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" || { echo "FAILED efficiency/$seed/$dataset/$model" >&2; return 1; }
  done
}
pids=(); for ((i=0; i<${#GPU_LIST[@]}*TASKS; i++)); do worker "${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" & pids+=("$!"); done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "Efficiency training failed; fix failed jobs before benchmarking." >&2; exit 1; }

# Benchmark each dataset separately: five models, same selected seed, same
# visible GPU, warm-up and timed-step policy. Do not compare rows if batch
# sizes differ; benchmark_models.py fails loudly in that case.
for dataset in cmumosei cmumosi iemocap; do
  mapfile -t configs < <(find "experiment1/generated/efficiency/seed_77/$dataset" -type f -name '*.ini' | sort)
  ((${#configs[@]} == 5)) || { echo "Missing efficiency configs for $dataset" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "${RUNNER[@]}" experiment1/benchmark_models.py \
    --configs "${configs[@]}" --require-same-batch --warmup "$WARMUP" --repeats "$REPEATS" \
    --output "$OUTPUT_ROOT/benchmarks/efficiency_seed77_${dataset}.csv"
done

# Sensitivity changes only the interaction-coordinate basis at test time;
# it never retrains or changes the checkpoint.
for dataset in cmumosei cmumosi iemocap; do
  config="experiment1/generated/main/seed_77/$dataset/qrsan.ini"
  checkpoint="$OUTPUT_ROOT/main/seed_77/$dataset/qrsan/best_model.pt"
  [[ -f "$config" && -f "$checkpoint" ]] || { echo "Missing seed-77 QRSAN checkpoint for $dataset; run main suite first." >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "${RUNNER[@]}" experiment1/basis_permutation_sensitivity.py \
    --config "$config" --model-file "$checkpoint" --output-dir "$OUTPUT_ROOT/basis_sensitivity/$dataset/seed_77" \
    --permutations "$PERMUTATIONS"
done

echo "Done. Efficiency CSVs: $OUTPUT_ROOT/benchmarks/; basis reports: $OUTPUT_ROOT/basis_sensitivity/."
