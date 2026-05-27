# MyDream Training Evaluation

Python tools for turning MyDream Android JSONL sleep exports into
training/evaluation tables and Colab/server-ready LightGBM inputs.

The current local role of this repository is preprocessing and packaging. Full
LightGBM training and repeated evaluation should run in Google Colab or a
server environment.

## Privacy

Do not commit personal sleep exports or generated outputs.

Ignored by default:

- `*.jsonl`
- `*.ndjson`
- `out/`
- `*.csv`
- `*.zip`
- model artifacts such as `*.onnx`, `*.tflite`, `*.joblib`, `*.pkl`

If tests need sample data, use only tiny anonymized fixtures under
`tests/fixtures/`.

## Current Files

- `parse_sleep_export.py`: converts Android JSONL exports into CSV tables
- `train_lightgbm_colab.py`: Colab/server LightGBM training/evaluation script
- `train_tabular_tflite_colab.py`: trains a small TFLite-friendly tabular model for Android runtime evaluation
- `analyze_alarm_failures.py`: compares threshold-level smart-alarm failure cases
- `build_sequence_dataset.py`: builds Phase 2 sleep-pattern sequence datasets
- `analyze_sequence_dataset.py`: inspects Phase 2 sequence dataset quality
- `train_sequence_model.py`: trains the first lightweight local sequence baseline
- `train_sequence_colab.py`: trains Colab/server GRU, CNN, CNN+GRU, TCN, or Transformer sequence models
- `run_sequence_experiment_matrix.py`: runs sequence architecture matrices and alarm-window comparisons
- `run_repeated_sequence_evaluation.py`: repeats selected sequence models across seeds and aggregates GRU deltas
- `build_combined_alarm_scores.py`: combines tabular, sequence, and deadline-closeness alarm scores for comparison
- `evaluate_decision_policies.py`: evaluates Android-compatible GRU/deadline/gate policies offline with coverage and utility metrics
- `convert_sequence_model_tflite.py`: converts the selected Keras sequence model to TensorFlow Lite and verifies sample outputs
- `mydream_lightgbm_colab.ipynb`: thin Colab runner for the training script
- `README.md`: local and Colab workflow notes

## Parse Export

Run without arguments to open a file picker:

```powershell
python parse_sleep_export.py
```

The default output directory is:

```text
out\<input-file-name>\
```

Or pass paths explicitly:

```powershell
python parse_sleep_export.py `
  --input C:\path\to\mydream_sleep_2026-05-18.jsonl `
  --out-dir out\verify_week_period_profile
```

Outputs:

- `sessions.csv`
- `stages.csv`
- `training_sessions.csv`
- `training_candidates_1min.csv`
- `alarm_candidates_1min.csv`
- `summary.json`

Canonical output directory for the current export:

```text
out\verify_week_period_profile\
```

## Default Assumptions

- Wake deadline is the user-defined `wake_time`, not `session.end`.
- Default historical wake-time policy is `month_day_of_week_profile`.
- Historical deadlines are estimated from each session's actual wake time in
  KST, using the median wake time for the same month and day of week.
- If a month/day-of-week cell has too few samples, the parser falls back to the
  same ISO week's weekday/weekend median, then to the overall weekday/weekend
  median.
- Fixed weekday/weekend wake times are still available for comparison with
  `--wake-time-policy fixed_weekday_weekend`, `--weekday-wake-time HH:MM`, and
  `--weekend-wake-time HH:MM`.
- The product target remains "must wake by 07:00" on weekdays for future data
  and runtime behavior.
- The weekend preference is currently treated as "wake between 08:00 and
  09:00", with `09:00` as the latest wake deadline. Because the smart alarm
  window is fixed at 30 minutes, weekend smart candidates fall in `08:30-09:00`.
- Search window is `[wake_time - 30 minutes, wake_time]`.
- Candidate interval is 1 minute.
- Training candidates are generated from the full recorded sleep-stage
  coverage, not only the wake-time alarm window.
- Alarm candidates are generated only inside `[wake_time - 30 minutes,
  wake_time]` for backtesting and smart-alarm selection.
- `wakeable` is positive when Awake or Light overlaps candidate time +/-5 minutes.
- `label_wakeable_window` keeps that +/-5 minute window label.
- `label_wakeable_at_candidate` is positive only when the candidate time itself
  is inside Awake or Light.
- `deep_soon` is positive when Deep starts within 10 minutes after candidate time
  and the candidate is not inside Deep.
- Candidates where the current stage is already Deep are excluded from training
  rows.
- Candidates where the current stage is `Unknown` are also excluded from
  training rows because they usually mean the candidate is outside recorded
  sleep-stage coverage.
- `summary.json` includes candidate coverage by session so we can see how many
  historical sessions still have usable candidates after excluding `Deep` and
  `Unknown`.
- Training candidates are generated only from sessions with stages and duration
  from 180 to 720 minutes.

## Local vs Colab Scope

Use the local PC for preprocessing, CSV generation, data-quality checks, small
baseline checks, and Colab/server input packaging.

Run the first full LightGBM training and repeated evaluation in Google Colab or
a server environment.

## Colab Inputs

Upload these files from the canonical output directory:

- `training_candidates_1min.csv`
- `alarm_candidates_1min.csv`
- `training_sessions.csv`
- `summary.json`

Optional:

- `sessions.csv`
- `stages.csv`
- original `mydream_sleep_2026-05-18.jsonl`

For convenience, the local workflow can package the Colab inputs into:

```text
out\verify_week_period_profile\mydream_colab_lightgbm_inputs.zip
```

## Leakage-Safe First Feature Set

Phase 1 tabular feature set:

- `minutes_before_deadline`
- `elapsed_sleep_minutes`
- `time_of_day_sin`
- `time_of_day_cos`
- `target_wake_hour_sin`
- `target_wake_hour_cos`
- `day_of_week`
- `stage_at_candidate`
- `previous_stage`
- `minutes_since_stage_start`
- `minutes_since_last_deep`
- `deep_cycle_position`
- `recent_30m_awake_minutes`
- `recent_30m_light_minutes`
- `recent_30m_deep_minutes`
- `recent_30m_rem_minutes`

Do not use:

- `month`
- `next_stage`
- `candidate_time`
- `deadline_time`
- `session_id`
- any `label_*` column

Use `--include-next-stage` only for an offline leakage comparison run. It is
future context and should not be used for app-realistic evaluation.

## Colab Training Script

Open `mydream_lightgbm_colab.ipynb` in Colab, or run
`train_lightgbm_colab.py` directly in Google Colab or a server environment.

Install dependencies in Colab:

```python
!pip install lightgbm pandas scikit-learn joblib
```

Run with a normal output directory:

```bash
python train_lightgbm_colab.py \
  --input-dir out/verify_week_period_profile \
  --output-dir out/verify_week_period_profile/model_eval
```

Run the leakage comparison separately if needed:

```bash
python train_lightgbm_colab.py \
  --input-dir out/verify_week_period_profile \
  --output-dir out/verify_week_period_profile/model_eval_with_next_stage \
  --include-next-stage
```

If the zip is unpacked directly into the current Colab working directory, run:

```bash
python train_lightgbm_colab.py \
  --input-dir . \
  --output-dir model_eval
```

## Tabular TFLite Runtime Candidate

Train a small deployment-oriented tabular model with the same leakage-safe
Phase 1 feature set:

```bash
python train_tabular_tflite_colab.py \
  --input-dir out/verify_week_period_profile \
  --output-dir out/verify_week_period_profile/model_tabular_tflite \
  --float16
```

Expected outputs:

- `model_tabular_tflite/tabular_model.keras`
- `model_tabular_tflite/tabular_model_float32.tflite`
- `model_tabular_tflite/tabular_model_float16.tflite`
- `model_tabular_tflite/tabular_feature_scaler.json`
- `model_tabular_tflite/tabular_tflite_manifest.json`
- `model_tabular_tflite/tabular_metrics.json`
- `model_tabular_tflite/tabular_predictions.csv`
- `model_tabular_tflite/alarm_predictions_long.csv`

Use `alarm_predictions_long.csv` as a drop-in tabular prediction file for
combined-score experiments:

```bash
python build_combined_alarm_scores.py \
  --tabular-predictions out/verify_week_period_profile/model_tabular_tflite/alarm_predictions_long.csv \
  --sequence-predictions out/verify_week_period_profile/sequence_experiments/gru/gru64_dense32_dropout00/alarm_predictions_long.csv \
  --output-dir out/verify_week_period_profile/combined_alarm_scores_tabular_tflite
```

Do not replace the LightGBM tabular baseline until this model's alarm-window
behavior is compared against `out/verify_week_period_profile/model_eval/`.

Expected outputs:

- `model_eval/label_deep_soon.joblib`
- `model_eval/label_deep_soon.lightgbm.txt`
- `model_eval/metrics.json`
- `model_eval/threshold_report.csv`
- `model_eval/test_predictions_long.csv`
- `model_eval/alarm_predictions_long.csv`
- `model_eval/prediction_sample.csv`
- `model_eval/session_top_candidates.csv`
- `model_eval/session_recommendation_summary.csv`
- `model_eval/alarm_backtest_summary.csv`

## Alarm Failure Analysis

After a model run, analyze threshold-level smart-alarm failures:

```bash
python analyze_alarm_failures.py \
  --model-dir out/verify_week_period_profile/model_eval
```

Compare two model runs by passing `--model-dir` twice:

```bash
python analyze_alarm_failures.py \
  --model-dir out/verify_week_period_profile/model_eval \
  --model-dir out/verify_week_period_profile/model_eval_second
```

Outputs are written to `failure_analysis/` by default:

- `threshold_comparison.csv`
- `model_comparison_summary.csv`
- `*_alarm_failure_cases_t0_4.csv`
- `*_alarm_failure_cases_t0_6.csv`
- `*_alarm_failure_cases_all_thresholds.csv`

## Data Quality And Fixed-Wake Coverage

Before model tuning, run the data-quality audit:

```bash
python audit_sleep_data_quality.py
```

Primary outputs:

```text
out/verify_week_period_profile/data_quality/data_quality_summary.json
out/verify_week_period_profile/data_quality/alarm_window_quality_by_session.csv
out/verify_week_period_profile/data_quality/fixed_wake_time_coverage_summary.csv
out/verify_week_period_profile/data_quality/fixed_wake_time_coverage_by_session.csv
out/verify_week_period_profile/data_quality/training_label_window_sensitivity.csv
out/verify_week_period_profile/data_quality/alarm_label_window_sensitivity.csv
out/verify_week_period_profile/data_quality/training_label_design_comparison.csv
out/verify_week_period_profile/data_quality/alarm_label_design_comparison.csv
```

Fixed wake-time sweep is a coverage stress test, not a direct model-accuracy verdict.

Current fixed wake-time coverage:

```text
05:00 evaluable 85.37% / evaluable-or-partial 86.79%
06:00 evaluable 82.32% / evaluable-or-partial 85.98%
07:00 evaluable 75.61% / evaluable-or-partial 77.85%
08:00 evaluable 53.66% / evaluable-or-partial 59.76%
09:00 evaluable 40.45% / evaluable-or-partial 42.68%
10:00 evaluable 25.61% / evaluable-or-partial 29.67%
11:00 evaluable 11.79% / evaluable-or-partial 13.62%
```

Interpretation:

- Use the full recorded sleep-stage coverage for sequence model training.
- Use historical wake-profile backtests as the primary model-development evaluation.
- Use fixed wake-time sweeps to understand coverage and product-policy stress.
- Compare model performance only on the same wake time and same coverage-qualified session set.
- Treat `coverage_rate >= 70%` as suitable for model comparison, `40-70%` as reference-only, and `<40%` as unsuitable for model-performance judgment.

Current label-design comparison:

```text
training all known candidates:
  deep_within_5m:  10,153 / 146,927 = 6.91%
  deep_within_10m: 18,973 / 146,927 = 12.91%
  deep_within_15m: 26,879 / 146,927 = 18.29%
  pre_entry_5_15m: 16,726 / 146,927 = 11.38%

alarm-window known candidates:
  deep_within_5m:     243 / 9,275 = 2.62%
  deep_within_10m:    447 / 9,275 = 4.82%
  deep_within_15m:    619 / 9,275 = 6.67%
  pre_entry_5_15m:    376 / 9,275 = 4.05%
```

Interpretation:

- Keep `deep_within_10m` as the primary Phase 2 target for now.
- `deep_within_5m` is likely too sparse as the only target.
- `deep_within_15m` may be useful as a broader risk target, but it can wake too early if used directly.
- `pre_entry_5_15m` is a promising secondary target because it focuses on a less-immediate Deep pre-entry region while keeping a usable positive ratio.

## Phase 2 Sequence Dataset

Build the first 60-minute sleep-pattern sequence dataset:

```bash
python build_sequence_dataset.py \
  --input-dir out/verify_week_period_profile
```

Default output:

- `out/verify_week_period_profile/sequence_60m/sequence_stage_ids.npy`
- `out/verify_week_period_profile/sequence_60m/sequence_metadata.csv`
- `out/verify_week_period_profile/sequence_60m/sequence_summary.json`
- `out/verify_week_period_profile/sequence_60m/stage_vocab.json`

Current sequence metadata includes:

```text
sequence_awake_ratio
sequence_light_ratio
sequence_deep_ratio
sequence_rem_ratio
sequence_unknown_ratio
sequence_stage_transition_count
sequence_known_stage_transition_count
```

These are used as context features by `train_sequence_colab.py` when present.

Inspect dataset quality:

```bash
python analyze_sequence_dataset.py \
  --sequence-dir out/verify_week_period_profile/sequence_60m
```

Quality outputs are written to `sequence_60m/quality/`:

- `sequence_quality_report.json`
- `sequence_split_summary.csv`
- `sequence_candidate_stage_label_summary.csv`
- `sequence_known_ratio_buckets.csv`
- `sequence_position_stage_ratios.csv`
- `sequence_transition_counts.csv`

Train the first local sequence baseline:

```bash
python train_sequence_model.py \
  --sequence-dir out/verify_week_period_profile/sequence_60m \
  --predict-sequence-dir out/verify_week_period_profile/sequence_60m_alarm \
  --output-dir out/verify_week_period_profile/sequence_60m/model_sequence_baseline
```

The local baseline uses flattened one-hot stage sequences plus metadata with
scikit-learn logistic regression. Use it as a sanity check before Colab/server
GRU or 1D CNN training, not as the final sequence model.

Compare the local sequence baseline against the Phase 1 tabular model:

```bash
python analyze_alarm_failures.py \
  --model-dir out/verify_week_period_profile/model_eval \
  --model-dir out/verify_week_period_profile/sequence_60m/model_sequence_baseline \
  --output-dir out/verify_week_period_profile/model_compare_tabular_vs_sequence
```

Train the real Phase 2 sequence model in Colab or a server environment:

```bash
python train_sequence_colab.py \
  --sequence-dir out/verify_week_period_profile/sequence_60m \
  --predict-sequence-dir out/verify_week_period_profile/sequence_60m_alarm \
  --output-dir out/verify_week_period_profile/sequence_60m/model_sequence_gru \
  --model-type gru
```

Run the 1D CNN comparison:

```bash
python train_sequence_colab.py \
  --sequence-dir out/verify_week_period_profile/sequence_60m \
  --predict-sequence-dir out/verify_week_period_profile/sequence_60m_alarm \
  --output-dir out/verify_week_period_profile/sequence_60m/model_sequence_cnn \
  --model-type cnn
```

Run the standalone GRU tuning matrix:

```bash
python run_sequence_experiment_matrix.py --experiment-set gru
```

The GRU matrix currently runs:

```text
GRU(32) + Dense(16) + Dropout(0.2)
GRU(64) + Dense(16) + Dropout(0.2)
GRU(32) + Dense(32) + Dropout(0.2)
GRU(64) + Dense(32) + Dropout(0.2)
GRU(64) + Dense(32) + Dropout(0.0)
```

Run the CNN+GRU expansion matrix only after standalone GRU tuning is understood:

```bash
python run_sequence_experiment_matrix.py --experiment-set cnn_gru
```

Run the Colab-first full sequence comparison workflow with:

```text
mydream_expanded_model_comparison_colab.ipynb
```

The notebook starts from a JSONL export stored on Google Drive and creates:

```text
stages.csv / training_candidates_1min.csv / alarm_candidates_1min.csv
sequence_60m/ / sequence_60m_alarm/
sequence_experiments/gru/gru64_dense32_dropout00/
sequence_experiments/expanded/
sequence_experiments/repeated_evaluation/
```

It then recomputes the deployment-relevant `0.55` threshold, displays
comparison tables/charts, and packages summary outputs. The initial notebook
configuration uses a fixed wake-time policy of weekday `07:00` and weekend
`09:00`; update those values in the settings cell when testing another policy.
By default `REUSE_EXISTING_RESULTS = False`, so models are retrained for the
configured input and policy. Set it to `True` only when resuming the same run.
The repeated evaluation section trains GRU, Transformer, and CNN+GRU with five
random seeds by default, so it runs 15 sequence-model training jobs.

Open directly in Colab:

```text
https://colab.research.google.com/github/sfpahsdev-mydream/mydream-training-evaluation/blob/main/mydream_expanded_model_comparison_colab.ipynb
```

Keep code in the runtime and persist input and generated results on Google
Drive:

```text
code root:    /content/mydream-training-evaluation
raw JSONL:    /content/drive/MyDrive/mydream_latest/input/mydream_sleep.jsonl
profile root: /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy
```

The notebook runs the following expanded command only after it has generated
the sequence datasets and the selected GRU result:

```bash
cd /content/mydream-training-evaluation
python run_sequence_experiment_matrix.py \
  --profile-root /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy \
  --experiment-set expanded \
  --comparison-model-dir /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy/sequence_experiments/gru/gru64_dense32_dropout00
```

`--profile-root` resolves `sequence_60m`, `sequence_60m_alarm`,
`sequence_experiments/gru/gru64_dense32_dropout00`, `model_tabular_tflite`,
and expanded `sequence_experiments` outputs under the supplied root. If no
tabular prediction has been produced for the same dataset yet, add
`--no-tabular-model` and compare sequence architectures first.

The `expanded` set trains:

```text
TCN(64) + Dense(32) + Dropout(0.0), dilations 1/2/4/8
Transformer(64, 4 heads, 2 blocks) + Dense(32) + Dropout(0.1)
CNN(32) + GRU(64) + Dense(32) + Dropout(0.0)
```

Repeat the deployment-threshold comparison across random seeds:

```bash
python run_repeated_sequence_evaluation.py \
  --profile-root /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy \
  --threshold 0.55 \
  --seed 42 --seed 43 --seed 44 --seed 45 --seed 46
```

Repeated evaluation outputs:

```text
sequence_experiments/repeated_evaluation/seed_<seed>/
sequence_experiments/repeated_evaluation/per_seed_summary.csv
sequence_experiments/repeated_evaluation/aggregate_summary.csv
sequence_experiments/repeated_evaluation/delta_vs_gru_per_seed.csv
sequence_experiments/repeated_evaluation/delta_vs_gru_summary.csv
```

Use `delta_vs_gru_summary.csv` to decide whether Transformer or CNN+GRU
consistently improves on GRU at threshold `0.55`.

Preview commands without training:

```bash
python run_sequence_experiment_matrix.py --experiment-set gru --dry-run
```

Current GRU tuning result:

```text
output:
  out/verify_week_period_profile/sequence_experiments/gru/

best raw GRU candidate:
  gru64_dense32_dropout00

threshold 0.4:
  smart 138 / fallback 354 / deep_success 45 / strong_fail 88

threshold 0.5:
  smart 116 / fallback 376 / deep_success 41 / strong_fail 71

threshold 0.6:
  smart 80 / fallback 412 / deep_success 34 / strong_fail 43
```

Next step: rerun combined scoring with
`out/verify_week_period_profile/sequence_experiments/gru/gru64_dense32_dropout00/alarm_predictions_long.csv`.

Current tuned GRU combined scoring result:

```text
output:
  out/verify_week_period_profile/combined_alarm_scores_gru64_dense32_dropout00/

best deployment-style candidate:
  combined_gru_50_tab_50 at threshold 0.55

result:
  smart 89 / fallback 403 / deep_success 39 / strong_fail 49
  success_per_smart 0.438202
```

Selected initial deployment candidate:

```text
model family:
  standalone GRU

model artifact:
  out/verify_week_period_profile/sequence_experiments/gru/gru64_dense32_dropout00/sequence_model.keras

scoring:
  combined_gru_50_tab_50
  threshold 0.55
```

Convert the selected model to TensorFlow Lite in Colab/server:

```bash
python convert_sequence_model_tflite.py --float16
```

Expected output:

```text
out/verify_week_period_profile/tflite/gru64_dense32_dropout00/
sequence_model_float32.tflite
sequence_model_float16.tflite
tflite_manifest.json
context_scaler.json
sequence_metrics.json
```

Current TFLite conversion result:

```text
float32 size: 275,956 bytes
float16 size: 243,420 bytes
rebuild max_abs_diff: 0.0
float32 max_abs_diff: 1.4901161193847656e-07
float32 mean_abs_diff: 1.1650932663087588e-08
```

Use `sequence_model_float32.tflite` first for Android output-parity validation. Use
`sequence_model_float16.tflite` after app-side validation passes.

Then compare against the tabular baseline:

```bash
python analyze_alarm_failures.py \
  --model-dir out/verify_week_period_profile/model_eval \
  --model-dir out/verify_week_period_profile/sequence_60m/model_sequence_gru \
  --model-dir out/verify_week_period_profile/sequence_60m/model_sequence_cnn \
  --output-dir out/verify_week_period_profile/model_compare_tabular_gru_cnn
```

Current Colab/server result summary:

```text
GRU test ROC-AUC: 0.903246
GRU test PR-AUC: 0.597314
GRU precision/recall at 0.5: 0.450520 / 0.814475

CNN test ROC-AUC: 0.896746
CNN test PR-AUC: 0.565252
CNN precision/recall at 0.5: 0.449347 / 0.800171
```

Current alarm-window interpretation:

- GRU is the best sequence result so far.
- GRU improves deep-success count versus tabular at both `0.4` and `0.6`.
- GRU also increases strong-fail count, so the next step is combined scoring rather than direct Android deployment.
- CNN is weaker than GRU for this dataset.

Build combined tabular + GRU + deadline-closeness score outputs:

```bash
python build_combined_alarm_scores.py
```

Compare the combined recipes:

```bash
python analyze_alarm_failures.py \
  --model-dir out/verify_week_period_profile/model_eval \
  --model-dir colab_result \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_50_tab_50 \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_60_tab_30_deadline_10 \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_60_tab_20_deadline_20 \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_70_tab_20_deadline_10 \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_60_tab_gate_0_4_deadline_10 \
  --output-dir out/verify_week_period_profile/combined_alarm_scores/comparison
```

Run the current threshold sweep:

```bash
python analyze_alarm_failures.py \
  --threshold 0.35 \
  --threshold 0.4 \
  --threshold 0.45 \
  --threshold 0.5 \
  --threshold 0.55 \
  --threshold 0.6 \
  --threshold 0.65 \
  --threshold 0.7 \
  --model-dir out/verify_week_period_profile/model_eval \
  --model-dir colab_result \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_50_tab_50 \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_60_tab_30_deadline_10 \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_60_tab_20_deadline_20 \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_70_tab_20_deadline_10 \
  --model-dir out/verify_week_period_profile/combined_alarm_scores/combined_gru_60_tab_gate_0_4_deadline_10 \
  --output-dir out/verify_week_period_profile/combined_alarm_scores/threshold_sweep
```

Current best combined tradeoff:

```text
combined_gru_60_tab_gate_0_4_deadline_10 at threshold 0.6:
  smart 84 / fallback 408 / deep_success 38 / strong_fail 45
```

## Offline Decision Policy Evaluation

Use desktop evaluation for policy selection and threshold sweeps. Keep Android
Lab evaluation focused on model/input parity, device latency, and a small
latest-session dry run.

Run the Android-compatible policy comparison for the current tuned GRU and
tabular TFLite prediction outputs:

```powershell
.\.venv\Scripts\python.exe evaluate_decision_policies.py `
  --threshold 0.55 `
  --output-dir out\verify_week_period_profile\policy_evaluation
```

Run a threshold sweep:

```powershell
.\.venv\Scripts\python.exe evaluate_decision_policies.py `
  --sweep `
  --output-dir out\verify_week_period_profile\policy_evaluation_sweep
```

Optional period filter:

```powershell
.\.venv\Scripts\python.exe evaluate_decision_policies.py `
  --recent-days 30 `
  --threshold 0.55 `
  --output-dir out\verify_week_period_profile\policy_evaluation_recent_30_days
```

The script evaluates these policies using the same score formulas and gate
limits as the Android Lab implementation:

- `GRU-only`
- `GRU + deadline`
- `GRU + strict deadline gate`
- `GRU + unknown coverage gate`
- `GRU + deadline + unknown gate`
- `GRU + tabular`

Inputs:

- GRU `alarm_predictions_long.csv`
- tabular `alarm_predictions_long.csv`
- `alarm_candidates_1min.csv`
- `sequence_60m_alarm/sequence_metadata.csv` for the 60-minute unknown ratio
- `stages.csv` to recompute coverage-aware `Deep within 10m` labels

Outputs:

- `policy_candidate_results.csv`: candidate-level policy decisions and outcome labels
- `policy_summary.csv`: coverage, precision/recall, false/missed smart, and utility summary
- `session_policy_summary.csv`: session-level utility results
- `threshold_sweep.csv`: written when multiple thresholds or `--sweep` are requested

`target_unknown` rows have incomplete or unknown stage coverage through the
10-minute target horizon. Candidates already in `Deep` are tracked as
`excluded_already_deep`, consistent with training-target exclusion. Both are
neutral in utility.

To reproduce an Android `recent_30_days_target_wake_policy` run exactly, the
input prediction and sequence metadata files must have been generated from the
same target-wake deadline candidate set used on device. `--recent-days 30`
only filters an already compatible candidate set; it does not change its wake
deadline policy.

## Evaluation Rules

- Split by session/date, not random candidate rows.
- Compare LightGBM against simple baselines.
- Phase 1 trains only `label_deep_soon` and outputs `P_deep_soon`.
- Track positive recall because `label_deep_soon` is expected to be highly
  imbalanced.
- Tune thresholds on validation rows, then read final quality on test rows.
- Backtest smart alarm selection before Android runtime work:
  - Select a smart alarm only before `wake_time`.
  - If no smart candidate is selected, use fallback at `wake_time`.
  - Report `selected_alarm_time`, `alarm_type`, `stage_at_alarm`,
    `next_deep_start`, `minutes_before_next_deep`,
    `deep_prevention_success`, `too_early_penalty`, `time_success`, and
    `sleep_quality_success` per session.
- Treat the current score draft as a starting point only:
  `P_deep_soon * (0.5 + 0.5 * deadline_urgency)`.

## Deferred Sequence Modeling

Do not implement sequence modeling until the Phase 1 tabular baseline and
backtest logic are validated.

Later sequence dataset:

- One row per candidate.
- Last 30-60 minutes at 1-minute resolution.
- Per-timestep features start with `sleep_stage`.
- Add heart rate, HRV, and motion after Wear OS log extraction is confirmed.

Later model candidates:

- Small GRU.
- Small LSTM.
- Small 1D CNN.

The final deployment target remains TensorFlow Lite, with sequence models kept
small enough for Android latency and model-size constraints.
