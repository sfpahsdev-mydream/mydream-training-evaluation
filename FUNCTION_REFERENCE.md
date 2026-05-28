# MyDream Training Evaluation Function Reference

This document explains what each training/evaluation script does, how to run it,
and which functions own each processing step.

Use `README.md` for the normal workflow. Use this file when changing code,
debugging an output, or mapping Android runtime behavior back to the Python
pipeline.

## Pipeline Map

```text
Android JSONL export
-> parse_sleep_export.py
-> training_candidates_1min.csv / alarm_candidates_1min.csv
-> data quality and label audits
-> LightGBM tabular baseline
-> sequence dataset
-> GRU/CNN sequence training
-> combined alarm scoring
-> TFLite conversion
-> Android parity samples
```

## Common Run Order

```powershell
.\.venv\Scripts\python.exe parse_sleep_export.py --input mydream_sleep_2026-05-18.jsonl --out-dir out\verify_week_period_profile
.\.venv\Scripts\python.exe audit_sleep_data_quality.py
.\.venv\Scripts\python.exe build_sequence_dataset.py --input-dir out\verify_week_period_profile
```

Colab/server-only TensorFlow or LightGBM steps:

```bash
python train_lightgbm_colab.py --input-dir out/verify_week_period_profile --output-dir out/verify_week_period_profile/model_eval
python train_sequence_colab.py --sequence-dir out/verify_week_period_profile/sequence_60m --predict-sequence-dir out/verify_week_period_profile/sequence_60m_alarm --output-dir out/verify_week_period_profile/sequence_experiments/gru/gru64_dense32_dropout00 --model-type gru --hidden-units 64 --dense-units 32 --dropout 0.0
python run_repeated_sequence_evaluation.py --profile-root /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy --threshold 0.4 --threshold 0.5 --threshold 0.55 --threshold 0.6 --seed 42 --seed 43 --seed 44 --seed 45 --seed 46 --evaluate-policies
python convert_sequence_model_tflite.py --float16
python train_tabular_tflite_colab.py --input-dir out/verify_week_period_profile --output-dir out/verify_week_period_profile/model_tabular_tflite --float16
```

## `parse_sleep_export.py`

Purpose: convert Android JSONL sleep exports into session, stage, training, alarm,
and summary CSV/JSON files.

Main usage:

```powershell
.\.venv\Scripts\python.exe parse_sleep_export.py --input mydream_sleep_2026-05-18.jsonl --out-dir out\verify_week_period_profile
```

Key classes:

- `Stage`: normalized sleep-stage interval.
- `Session`: normalized sleep session with nested stages.
- `WakeProfile`: historical wake-time profile used to choose offline deadlines.

Key functions:

- `load_sessions`, `parse_session`, `validate_session`: read and normalize Android JSONL records.
- `candidate_times`, `session_candidate_times`: generate 1-minute candidate timestamps.
- `build_wake_profile`, `wake_time_for_session`, `fixed_wake_time_for_session`: choose deadline policy.
- `stage_at_time`, `stage_index_at_time`, `stage_context`: compute current, previous, and next stage context.
- `recent_stage_minutes`, `minutes_since_last_deep`, `deep_cycle_position`: generate tabular/sequence context features.
- `label_wakeable`, `label_wakeable_at_candidate`, `label_deep_soon`, `next_deep_start`: build training/evaluation labels.
- `write_sessions_csv`, `write_stages_csv`, `write_candidates_csv`, `write_summary`: write pipeline outputs.
- `summarize_candidates`, `summarize_candidate_session_coverage`, `build_summary`: produce sanity-check metadata.

Important outputs:

- `sessions.csv`
- `stages.csv`
- `training_sessions.csv`
- `training_candidates_1min.csv`
- `alarm_candidates_1min.csv`
- `summary.json`

## `audit_sleep_data_quality.py`

Purpose: inspect data coverage, label design, wake-window coverage, and suspicious
stage continuity before tuning models.

Main usage:

```powershell
.\.venv\Scripts\python.exe audit_sleep_data_quality.py
```

Key functions:

- `add_session_times`: attaches parsed session timing columns.
- `audit_stage_continuity`: finds gaps, overlaps, and abnormal stage sequences.
- `audit_alarm_window`: checks candidate availability in the alarm window.
- `audit_fixed_wake_coverage`: evaluates fixed wake-time coverage stress tests.
- `audit_label_windows`, `audit_label_design`: compare Deep-window label variants.
- `stage_distribution`, `summarize`: write high-level quality reports.

Important outputs live under `out/verify_week_period_profile/data_quality/`.

## `train_lightgbm_colab.py`

Purpose: train the Phase 1 LightGBM tabular baseline for `label_deep_soon`.

Main usage:

```bash
python train_lightgbm_colab.py --input-dir out/verify_week_period_profile --output-dir out/verify_week_period_profile/model_eval
```

Key functions:

- `chronological_session_split`: split by session date, not random candidate rows.
- `feature_columns`, `build_feature_matrix`, `align_feature_matrix`: create leakage-safe one-hot tabular features.
- `train_target`: train one LightGBM model and write metrics/importance.
- `predict_target`: score train/test or alarm-window candidates.
- `build_session_top_candidates`, `build_session_recommendation_summary`: inspect top model candidates per session.
- `build_alarm_backtest_summary`, `alarm_backtest_metrics`: convert candidate probabilities into alarm-window outcome metrics.
- `threshold_metrics`: summarize precision/recall across thresholds.

Important outputs:

- `model_eval/label_deep_soon.joblib`
- `model_eval/alarm_predictions_long.csv`
- `model_eval/alarm_backtest_summary.csv`
- `model_eval/threshold_report.csv`
- `model_eval/metrics.json`

## `train_tabular_tflite_colab.py`

Purpose: train a small Keras MLP tabular model that can run on Android/Wear OS as
TensorFlow Lite.

Main usage:

```bash
python train_tabular_tflite_colab.py --input-dir out/verify_week_period_profile --output-dir out/verify_week_period_profile/model_tabular_tflite --float16
```

Key functions:

- `build_feature_matrix`, `align_feature_matrix`: use the same Phase 1 feature columns as LightGBM.
- `standardize_features`: produce `tabular_feature_scaler.json` for Android.
- `build_model`: create the small tabular MLP.
- `convert_float32`, `convert_float16`: write TFLite models.
- `run_tflite`, `verify_tflite`: compare Keras and TFLite outputs.
- `prediction_rows`, `threshold_report`: write compatible prediction/evaluation CSVs.

Important outputs:

- `tabular_model_float32.tflite`
- `tabular_model_float16.tflite`
- `tabular_feature_scaler.json`
- `tabular_tflite_manifest.json`
- `alarm_predictions_long.csv`

## `build_sequence_dataset.py`

Purpose: turn candidate rows into 60-minute sequence model inputs.

Main usage:

```powershell
.\.venv\Scripts\python.exe build_sequence_dataset.py --input-dir out\verify_week_period_profile
```

Key functions:

- `chronological_session_splits`: keep train/validation/test split aligned by session.
- `build_session_stage_grid`: create per-minute stage arrays for each session.
- `build_sequences`: generate `sequence_stage_ids.npy` and metadata rows.
- `add_sequence_context_features`: add ratios, transition counts, and known/unknown coverage features.
- `sequence_summary`: write dataset-level counts and label ratios.

Important outputs:

- `sequence_60m/sequence_stage_ids.npy`
- `sequence_60m/sequence_metadata.csv`
- `sequence_60m/stage_vocab.json`
- `sequence_60m/sequence_summary.json`
- `sequence_60m_alarm/...`

## `analyze_sequence_dataset.py`

Purpose: inspect sequence dataset quality before training GRU/CNN models.

Main usage:

```powershell
.\.venv\Scripts\python.exe analyze_sequence_dataset.py --sequence-dir out\verify_week_period_profile\sequence_60m
```

Key functions:

- `split_summary`: counts and label ratios by split.
- `candidate_stage_label_summary`: label rate by candidate stage.
- `known_ratio_buckets`: Unknown/known sequence coverage buckets.
- `position_stage_ratios`: stage distribution by minute position in the 60-minute window.
- `transition_counts`: local transition-density checks.
- `end_stage_match_summary`: verifies sequence end aligns with candidate stage.

## `train_sequence_model.py`

Purpose: train a lightweight local sequence sanity-check model without TensorFlow.

Main usage:

```powershell
.\.venv\Scripts\python.exe train_sequence_model.py --sequence-dir out\verify_week_period_profile\sequence_60m --predict-sequence-dir out\verify_week_period_profile\sequence_60m_alarm --output-dir out\verify_week_period_profile\sequence_60m\model_sequence_baseline
```

Key functions:

- `one_hot_sequences`: flatten stage sequences into one-hot features.
- `build_features`, `build_scoring_features`: combine sequence and metadata features.
- `build_pipeline`: create the scikit-learn baseline pipeline.
- `prediction_frame`, `alarm_prediction_frame`: write compatible prediction CSVs.

## `train_sequence_colab.py`

Purpose: train the real Phase 2 sequence model in TensorFlow: GRU, LSTM, CNN,
CNN+GRU, TCN, or a small Transformer encoder.

Main usage:

```bash
python train_sequence_colab.py --sequence-dir out/verify_week_period_profile/sequence_60m --predict-sequence-dir out/verify_week_period_profile/sequence_60m_alarm --output-dir out/verify_week_period_profile/sequence_experiments/gru/gru64_dense32_dropout00 --model-type gru --hidden-units 64 --dense-units 32 --dropout 0.0
```

Key functions:

- `load_dataset`: read `sequence_metadata.csv` and `sequence_stage_ids.npy`.
- `standardize_context`: create `context_scaler.json`.
- `build_model`: construct the selected sequence architecture, including
  causal dilated TCN residual blocks and Transformer attention blocks.
- `metrics_for_split`, `threshold_report`: evaluate train/validation/test behavior.
- `prediction_rows`: write compatible prediction CSVs for analysis.

Important outputs:

- `sequence_model.keras`
- `context_scaler.json`
- `sequence_metrics.json`
- `alarm_predictions_long.csv`

## `run_sequence_experiment_matrix.py`

Purpose: run predefined GRU, CNN+GRU, post-GRU expanded, large-capacity, or
advanced architecture grids.

`mydream_expanded_model_comparison_colab.ipynb` is the end-to-end Colab runner.
It creates preprocessing output, `sequence_60m`, `sequence_60m_alarm`, and
`sequence_experiments/gru/gru64_dense32_dropout00` before invoking the
configured matrix. The notebook now defaults to the `all` matrix and filters
the actual run with model-level `EXPERIMENT_MODEL_FLAGS`.

Main usage:

```powershell
.\.venv\Scripts\python.exe run_sequence_experiment_matrix.py --experiment-set gru --dry-run
```

Expanded comparison against the already trained selected GRU at
`sequence_experiments/gru/gru64_dense32_dropout00`:

```bash
cd /content/mydream-training-evaluation
python run_sequence_experiment_matrix.py --profile-root /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy --experiment-set expanded --comparison-model-dir /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy/sequence_experiments/gru/gru64_dense32_dropout00
```

Large-capacity Colab comparison after Android latency was found to be well
inside the 1-second budget:

```bash
cd /content/mydream-training-evaluation
python run_sequence_experiment_matrix.py --profile-root /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy --experiment-set large --comparison-model-dir /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy/sequence_experiments/gru/gru64_dense32_dropout00
```

Run the full registered set but train only selected models:

```bash
cd /content/mydream-training-evaluation
python run_sequence_experiment_matrix.py --profile-root /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy --experiment-set all --include-experiment gru256_dense128_dropout10 --include-experiment bigru_attention128_dense128_dropout10 --comparison-model-dir /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy/sequence_experiments/gru/gru64_dense32_dropout00
```

Key functions:

- `selected_experiments`: select GRU, CNN+GRU, expanded, large, advanced, or all experiment definitions.
- `filter_experiments`: apply `--include-experiment` and `--exclude-experiment` model-level filters.
- `parse_args`: resolve `--profile-root` into the standard profile dataset and output folders.
- `train_command`: build the `train_sequence_colab.py` command.
- `analyze_command`: build the comparison command after predictions exist.
- `run_command`: execute or print commands.

## `run_repeated_sequence_evaluation.py`

Purpose: repeat packaged, large-capacity, advanced, or all sequence-model
training across random seeds and thresholds, aggregate comparison against the
GRU reference, and optionally compare the same candidate-level policies for
each model.

Main usage:

```bash
python run_repeated_sequence_evaluation.py --profile-root /content/drive/MyDrive/mydream_latest/out/latest_fixed_wake_policy --experiment-set all --include-experiment gru64_dense32_dropout00 --include-experiment gru256_dense128_dropout10 --include-experiment bigru_attention128_dense128_dropout10 --threshold 0.4 --threshold 0.5 --threshold 0.55 --threshold 0.6 --seed 42 --seed 43 --seed 44 --seed 45 --seed 46 --evaluate-policies
```

Important outputs:

- `sequence_experiments/repeated_evaluation_all/per_seed_summary.csv`
- `sequence_experiments/repeated_evaluation_all/aggregate_summary.csv`
- `sequence_experiments/repeated_evaluation_all/delta_vs_gru_per_seed.csv`
- `sequence_experiments/repeated_evaluation_all/delta_vs_gru_summary.csv`
- `sequence_experiments/repeated_evaluation_all/policy_per_seed_summary.csv`
- `sequence_experiments/repeated_evaluation_all/policy_aggregate_summary.csv`

## `convert_sequence_model_tflite.py`

Purpose: convert the selected Keras GRU model into TFLite and verify output parity.

Main usage:

```bash
python convert_sequence_model_tflite.py --float16
```

Key functions:

- `load_context_scaler`, `load_sample`: read conversion metadata and verification samples.
- `rebuild_unrolled_gru_model`: rebuild GRU with `unroll=True` for cleaner TFLite conversion.
- `convert_float32`, `convert_float16`: write TFLite artifacts.
- `run_tflite`, `verify_model`: compare Keras and TFLite outputs.

Important outputs:

- `sequence_model_float32.tflite`
- `sequence_model_float16.tflite`
- `context_scaler.json`
- `tflite_manifest.json`

## `build_combined_alarm_scores.py`

Purpose: combine tabular, sequence, and deadline-closeness prediction files into
compatible alarm prediction directories.

Main usage:

```powershell
.\.venv\Scripts\python.exe build_combined_alarm_scores.py --tabular-predictions out\verify_week_period_profile\model_tabular_tflite\alarm_predictions_long.csv --sequence-predictions out\verify_week_period_profile\sequence_experiments\gru\gru64_dense32_dropout00\alarm_predictions_long.csv --output-dir out\verify_week_period_profile\combined_alarm_scores_tabular_tflite
```

Key functions:

- `read_predictions`: load and normalize probability columns.
- `deadline_closeness`: compute urgency inside the 30-minute alarm window.
- `merge_predictions`: align tabular and sequence rows by session/candidate/deadline.
- `apply_recipe`: apply score recipes such as `combined_gru_50_tab_50`.
- `recipe_metadata`: write recipe details for reproducibility.

## `analyze_alarm_failures.py`

Purpose: summarize smart-alarm behavior and failure cases for one or more model
directories.

Main usage:

```powershell
.\.venv\Scripts\python.exe analyze_alarm_failures.py --model-dir out\verify_week_period_profile\model_eval --model-dir out\verify_week_period_profile\model_tabular_tflite --output-dir out\verify_week_period_profile\model_compare_lightgbm_vs_tabular_tflite
```

Key functions:

- `read_alarm_predictions`: load `alarm_predictions_long.csv`.
- `select_alarm`: select the first/eligible smart candidate or fallback.
- `too_early_penalty`: classify very early wake choices.
- `opportunity_summary`: count sessions with useful alarm opportunities.
- `add_failure_classification`, `summarize_cases`: create failure-case CSVs.

Important outputs:

- `threshold_comparison.csv`
- `model_comparison_summary.csv`
- `*_alarm_failure_cases_*.csv`

## `evaluate_decision_policies.py`

Purpose: evaluate Android-compatible decision policies offline from one or
more saved sequence-model prediction outputs and one tabular prediction output
with coverage-aware labels and utility metrics.

Main usage:

```powershell
.\.venv\Scripts\python.exe evaluate_decision_policies.py --profile-root out\latest_fixed_wake_policy --sequence-prediction GRU=out\latest_fixed_wake_policy\sequence_experiments\gru\gru64_dense32_dropout00\alarm_predictions_long.csv --sequence-prediction TCN=out\latest_fixed_wake_policy\sequence_experiments\expanded\tcn64_dense32_dropout00\alarm_predictions_long.csv --sequence-prediction CNN+GRU=out\latest_fixed_wake_policy\sequence_experiments\expanded\cnn32_gru64_dense32_dropout00\alarm_predictions_long.csv --threshold 0.55 --output-dir out\latest_fixed_wake_policy\policy_evaluation
```

Threshold sweep:

```powershell
.\.venv\Scripts\python.exe evaluate_decision_policies.py --sweep --output-dir out\verify_week_period_profile\policy_evaluation_sweep
```

Key functions:

- `merge_inputs`: align each sequence model, tabular, candidate, and sequence metadata rows.
- `actual_label`: recompute coverage-aware `label_deep_soon` from `stages.csv`.
- `policy_decision`: apply comparable sequence/deadline/gate/tabular score logic.
- `summarize_policy`: calculate coverage, precision/recall, false/missed smart, and utility.
- `session_summary`: export per-session utility distributions.

Important outputs:

- `policy_candidate_results.csv`
- `policy_summary.csv`
- `session_policy_summary.csv`
- `threshold_sweep.csv` when a sweep is run

Important constraint:

- Use prediction files built from the same deadline candidate policy when
  comparing results with an Android Lab run. The `--recent-days` option filters
  rows only; it does not regenerate fixed target-wake candidates.

## `create_android_parity_sample.py`

Purpose: create Android asset JSON files used to validate app-side GRU and
tabular TFLite output parity.

Main usage:

```powershell
.\.venv\Scripts\python.exe create_android_parity_sample.py --sequence-dir out\verify_week_period_profile\sequence_60m_alarm --prediction-csv out\verify_week_period_profile\sequence_experiments\gru\gru64_dense32_dropout00\alarm_predictions_long.csv --tabular-prediction-csv out\verify_week_period_profile\model_tabular_tflite\alarm_predictions_long.csv --tabular-candidates-csv out\verify_week_period_profile\alarm_candidates_1min.csv --tabular-scaler-json out\verify_week_period_profile\model_tabular_tflite\tabular_feature_scaler.json --scaler-json out\verify_week_period_profile\tflite\gru64_dense32_dropout00\context_scaler.json --output ..\mydream-android\feature\on-device-inference\src\main\assets\mydream_sequence_gru64_dense32_dropout00\parity_sample.json --multi-output ..\mydream-android\feature\on-device-inference\src\main\assets\mydream_sequence_gru64_dense32_dropout00\parity_samples.json
```

Key functions:

- `build_sample`: assemble one fixed Android parity sample.
- `find_matching_prediction`: align sequence, tabular, and candidate rows.
- `build_tabular_features`: generate raw/scaled 28-column tabular inputs.
- `select_multi_sample_indices`: select low, mid, threshold, high, Unknown, and positive-label samples.

Important Android asset fields:

- `stage_sequence_60m`
- `context_raw_22`
- `context_scaled_22`
- `expected_gru_score`
- `tabular_raw_28`
- `tabular_scaled_28`
- `expected_tabular_score`
