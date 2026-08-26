# Experiment 1 supplementary experiments

`run_experiment1_linux.sh` runs the registered QRSAN main, available ablation and efficiency suites with five matched seeds, then writes prediction, IEMOCAP split and reproducibility audits. It does not claim that the architecture variants listed in `manifest.json` as `required_but_not_implemented` have been run.

To run both Section 4.3 and Experiment 1 through one GPU-safe entry point, use `run_all_reviewer_experiments_linux.sh`.

```bash
bash experiment1/run_experiment1_linux.sh --gpus 0,1 --conda-env multimodal-sa
```

To regenerate audits from existing artifacts only:

```bash
bash experiment1/run_experiment1_linux.sh --analyze-only --conda-env multimodal-sa
```

The full run additionally writes `eval/experiment1/benchmarks/efficiency_seed77.csv` and ten fixed-model basis permutations per dataset under `eval/experiment1/basis_sensitivity/`. The benchmark records parameter count, training-step time, inference latency and peak allocated GPU memory after warm-up.

Create IEMOCAP leave-one-session-out manifests only from real source metadata:

```bash
python experiment1/make_iemocap_loso_folds.py \
  --metadata-csv path/to/iemocap_utterances.csv \
  --output-dir eval/experiment1/iemocap_loso_folds
```

The CSV must provide `utterance_id`, `session`, and `speaker`. The current prepared pickle does not supply those fields, so no script may infer them.

Create the parameter-matched real-valued configuration before launching its five matched seeds. The candidate grid must be widened if no candidate is within the declared 1% tolerance.

```bash
python experiment1/parameter_match.py \
  --target-config config/reproduction/qrsan.ini \
  --candidate-dims 8,8,8 9,9,9 10,10,10 11,11,11 12,12,12 \
  --output-config experiment1/generated/real_qrsan_matched.ini \
  --report eval/experiment1/parameter_matching/real_qrsan.json
```

Before using IEMOCAP results, inspect `eval/experiment1/audits/iemocap/iemocap_split_audit.json`. A prepared pickle without original utterance IDs, speakers and sessions cannot establish a speaker-disjoint protocol; it must not be described as such.
