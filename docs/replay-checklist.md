# Frozen replay verification

We compare the frozen Test 2 replay with the saved static predictions and use
synthetic examples to test the verifier.
Waiting for delayed-label replay to be completed so we can check it.

## What we check

- Use only Test 2 and a saved source checkpoint; keep all weights fixed.
- Identify a forecast by `(run, bearing, input_index)`, not input index alone.
- Check calibration, history, timestamps, and target values against the data rules
  in [data_docs/README.md](../data_docs/README.md).
- Match the scored pairs and predictions to the static reference within `1e-12`.
- Report overall and per-bearing MAE/RMSE in log10(RMS / 1 g) units. Pool errors
  for the overall metrics. Late-life MAE uses Test 2 B1 targets in the final 10%
  of elapsed live-run time. Include `n_late`; an empty late subset is unavailable.

## Draft event CSV (synthetic tests)

Each issued forecast has one `predict` row and one `release` or `discard` row.
Within each bearing, keep events chronological and predict before releasing
labels at the same timestamp. Use the original observation timestamps.

| Columns | Meaning |
|---|---|
| `event_id` | Consecutive row number starting at zero. |
| `event` | `predict`, `release`, or `discard`. |
| `run`, `bearing`, `input_index` | Forecast identity; run is 2. |
| `time`, `mlp` | Prediction timestamp and fixed predicted value, repeated on its terminal row. |
| `event_time`, `observation_index` | Current stream observation when the event occurs; zero-based index into the untrimmed raw table. |
| `target_index`, `target_time`, `y` | Actual observed label; populated only on `release`. Blank on prediction/discard. |
| `eligible_for_training` | Whether a valid pair's label is now available; false on prediction/discard. |
| `eligible_for_scoring` | Whether the released pair belongs to the existing static evaluation subset. |
| `reason` | Blank except on discard: `gap`, `horizon`, or `end_of_stream`. |

Release a pair at its sixth later observation only when all intervals stay within
9–11 minutes and the total horizon is 60 ± 1 minutes. Discard it at the first
intervening gap, an invalid horizon, or the last observation if no target arrives.

The draft stream includes shutdown observations and predictions whose targets
later turn out to be invalid. We apply shutdown removal when scoring the saved
predictions. `eligible_for_scoring` is filled in after the run;
`eligible_for_training` only means that the label has arrived. We still need to
decide how to handle shutdown observations during training.

## Run the independent checks

After the README's setup and static reproduction, run from the repository root:

```powershell
.\.venv\Scripts\python.exe code\online_replay.py --data-dir data --checkpoint runs\local\results\static_mlp\model.npz --reference-predictions runs\local\results\static_mlp\test2_predictions.csv --out runs\verification\replay
.\.venv\Scripts\python.exe scripts\check_replay.py --data-dir data --checkpoint runs\local\results\static_mlp\model.npz --predictions runs\verification\replay\test2_predictions.csv --reference-predictions runs\local\results\static_mlp\test2_predictions.csv --out runs\verification\checks
```

The verifier runs both sets of checks and saves `summary.json` and `metrics.csv`.
We keep the latest reviewed copies in `results/replay_verification/`; predictions
and other run files stay under `runs/`. Unavailable late MAEs are blank.

Synthetic tests check early labels, mixed bearings, missing or duplicate events,
gaps, and changed forecasts or weights. The real-data check runs the frozen model
and compares its weights before and after each prediction. Test 3 is not accessed.

To run only the synthetic tests, use `python scripts/test_check_replay.py`.
The optional `--events path\to\events.csv` argument checks a saved event log.
