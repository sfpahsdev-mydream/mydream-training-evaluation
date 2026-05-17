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
  --input C:\path\to\mydream_sleep_2026-05-17.jsonl `
  --out-dir out\mydream_sleep_2026-05-17
```

Outputs:

- `sessions.csv`
- `stages.csv`
- `training_sessions.csv`
- `candidates_5min.csv`
- `summary.json`

Canonical output directory for the current export:

```text
out\mydream_sleep_2026-05-17\
```

## Default Assumptions

- Wake deadline is temporarily `session.end`.
- Search window is 30 minutes before deadline.
- Candidate interval is 5 minutes.
- `wakeable` is positive when Awake or Light overlaps candidate time +/-5 minutes.
- `label_wakeable_window` keeps that +/-5 minute window label.
- `label_wakeable_at_candidate` is positive only when the candidate time itself
  is inside Awake or Light.
- `deep_soon` is positive when Deep starts within 10 minutes after candidate time
  and the candidate is not inside Deep.
- Training candidates are generated only from sessions with stages and duration
  from 180 to 720 minutes.

## Local vs Colab Scope

Use the local PC for preprocessing, CSV generation, data-quality checks, small
baseline checks, and Colab/server input packaging.

Run the first full LightGBM training and repeated evaluation in Google Colab or
a server environment.

## Colab Inputs

Upload these files from the canonical output directory:

- `candidates_5min.csv`
- `training_sessions.csv`
- `summary.json`

Optional:

- `sessions.csv`
- `stages.csv`
- original `mydream_sleep_2026-05-17.jsonl`

For convenience, the local workflow can package the Colab inputs into:

```text
out\mydream_sleep_2026-05-17\mydream_colab_lightgbm_inputs.zip
```

## Leakage-Safe First Feature Set

Use:

- `minutes_before_deadline`
- `elapsed_sleep_minutes`
- `stage_at_candidate`
- `previous_stage`
- `minutes_since_stage_start`
- `recent_30m_awake_minutes`
- `recent_30m_light_minutes`
- `recent_30m_deep_minutes`
- `recent_30m_rem_minutes`
- `recent_30m_unknown_minutes`

Do not use:

- `next_stage`
- `candidate_time`
- `deadline_time`
- `session_id`
- any `label_*` column

## Colab Training Script

`train_lightgbm_colab.py` is intended to run in Google Colab or a server
environment.

Install dependencies in Colab:

```python
!pip install lightgbm pandas scikit-learn
```

Run with a normal output directory:

```bash
python train_lightgbm_colab.py \
  --input-dir out/mydream_sleep_2026-05-17 \
  --output-dir out/mydream_sleep_2026-05-17/model_eval
```

If the zip is unpacked directly into the current Colab working directory, run:

```bash
python train_lightgbm_colab.py \
  --input-dir . \
  --output-dir model_eval
```

Expected outputs:

- `model_eval/model_eval_summary.json`
- `model_eval/predictions.csv`

## Evaluation Rules

- Split by session/date, not random candidate rows.
- Compare LightGBM against simple baselines.
- Use `label_wakeable_at_candidate` as the first `P_wakeable` target.
- Use `label_deep_soon` as the first `P_deep_soon` target.
- Keep future-information columns such as `next_stage` out of the first feature
  set.
