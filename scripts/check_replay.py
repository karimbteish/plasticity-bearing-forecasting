"""Verify frozen Test 2 forecasts; optionally check the proposed event log.

Eligibility and metrics are recomputed independently of baseline.make_examples
and baseline.scores. Runtime weight checks exercise the existing replay function.
See docs/replay-checklist.md for the event format and limits of these checks.
"""
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import platform
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import online_replay
from baseline import load_run, make_examples
from static_mlp import Adam, MLP

KEYS = ["run", "bearing", "input_index"]
ATOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def flags(values):
    """Reject missing or ambiguous booleans rather than treating strings as true."""
    result = values.astype(str).str.lower().map({"true": True, "false": False})
    require(result.notna().all(), f"Invalid boolean in {values.name}")
    return result.astype(bool)


def integers(frame, columns):
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        require(np.isfinite(values).all() and (values == np.floor(values)).all(),
                f"Invalid integer in {column}")
        frame[column] = values.astype(np.int64)


def read_observations(data_dir):
    """Read only Test 2 raw features, including its trailing shutdown observations."""
    raw = pd.read_csv(data_dir / "test2_raw.csv")
    times = pd.to_datetime(raw.timestamp, errors="raise")
    require(times.notna().all() and times.is_unique and times.is_monotonic_increasing,
            "Observation timestamps must be unique and increasing")
    require(raw.file_index.is_unique, "Duplicate raw file_index")
    rms = raw[[f"Bearing{b}_ch1_rms" for b in range(1, 5)]]
    require(len(raw) > 206 and np.isfinite(rms).all().all() and (rms > 0).all().all(),
            "Need positive, finite RMS values and enough calibration/history")
    quiet = (rms < .2 * rms.iloc[:200].median()).all(axis=1)
    live_end = len(raw)
    while live_end and quiet.iloc[live_end - 1]:
        live_end -= 1
    return pd.concat([
        pd.DataFrame(dict(run=2, bearing=b, observation_index=np.arange(len(raw)),
                          time=times, y=np.log10(rms.iloc[:, b-1]),
                          live=np.arange(len(raw)) < live_end))
        for b in range(1, 5)
    ], ignore_index=True)


def pair_rules(observations):
    """An independent offline oracle, never supplied to prediction code.

Predict after calibration with six observations of history. A pair ends at its
sixth later observation, the first intervening gap, or the end of the stream.
"""
    rows = []
    for (run, bearing), group in observations.groupby(["run", "bearing"], sort=False):
        group = group.reset_index(drop=True)
        require(run == 2 and bearing in (1, 2, 3, 4), "Only Test 2 bearings are supported")
        require(np.array_equal(group.observation_index, np.arange(len(group))),
                "Observation indices must be contiguous from zero within each bearing")
        t = pd.to_datetime(group.time)
        require(t.notna().all() and t.is_unique and t.is_monotonic_increasing,
                "Observation timestamps must be unique and increasing")
        live = flags(group.live)
        require(live.any() and not ((~live).cummax() & live).any(), "Live rows must form a prefix")
        late_start = t.iloc[0] + .9 * (t[live].iloc[-1] - t.iloc[0])
        regular = t.diff().dt.total_seconds().between(540, 660)
        segment = (~regular).cumsum().to_numpy()
        history = group.groupby(segment).cumcount().to_numpy() + 1
        for i in range(200, len(group)):
            if history[i] < 6:
                continue
            end = min(i + 6, len(group) - 1)
            gaps = np.flatnonzero(segment[i+1:end+1] != segment[i])
            if gaps.size:
                end, event, reason = i + 1 + int(gaps[0]), "discard", "gap"
            elif i + 6 >= len(group):
                event, reason = "discard", "end_of_stream"
            elif abs((t.iloc[end] - t.iloc[i]).total_seconds() - 3600) > 60:
                event, reason = "discard", "horizon"
            else:
                event, reason = "release", ""
            released = event == "release"
            scored = released and bool(live.iloc[i] and live.iloc[end])
            rows.append(dict(run=int(run), bearing=int(bearing), input_index=i,
                             time=t.iloc[i], event=event, event_time=t.iloc[end],
                             observation_index=end, reason=reason,
                             target_index=end if released else None,
                             target_time=t.iloc[end] if released else pd.NaT,
                             y=float(group.y.iloc[end]) if released else None,
                             eligible_for_training=released, eligible_for_scoring=scored,
                             late=scored and bearing == 1 and t.iloc[end] >= late_start))
    require(bool(rows), "No eligible prediction inputs")
    return pd.DataFrame(rows)


def validate_predictions(frame, rules):
    frame = frame.copy()
    needed = KEYS + ["target_index", "time", "target_time", "y", "mlp", "late"]
    require(set(needed) <= set(frame), f"Missing prediction columns: {set(needed) - set(frame)}")
    integers(frame, KEYS + ["target_index"])
    for name in ("time", "target_time"):
        frame[name] = pd.to_datetime(frame[name], errors="raise")
    frame["late"] = flags(frame.late)
    require(not frame.duplicated(KEYS).any(), "Duplicate prediction identity")
    expected = rules.loc[rules.eligible_for_scoring].sort_values(KEYS).reset_index(drop=True)
    actual = frame.sort_values(KEYS).reset_index(drop=True)
    require(len(actual) == len(expected) and len(actual) > 0, "Scored pair count differs")
    for name in KEYS + ["target_index", "time", "target_time", "late"]:
        require((actual[name] == expected[name]).all(), f"Prediction {name} differs from observations")
    require(np.isfinite(actual[["y", "mlp"]]).all().all(), "Nonfinite prediction or target")
    require(np.allclose(actual.y, expected.y, rtol=0, atol=ATOL), "Targets differ from observed RMS")
    return actual


def metric_rows(predictions):
    """Point-weighted MAE/RMSE in log10(RMS / 1 g); no mean of bearing RMSEs."""
    rows = []
    for bearing in ("all", 1, 2, 3, 4):
        group = predictions if bearing == "all" else predictions.loc[predictions.bearing == bearing]
        require(len(group) > 0, f"No scored forecasts for bearing {bearing}")
        error = group.mlp.to_numpy() - group.y.to_numpy()
        late = flags(group.late).to_numpy()
        rows.append(dict(bearing=bearing, n=len(group), n_late=int(late.sum()),
                         mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                         late_mae=float(np.mean(np.abs(error[late]))) if late.any() else None))
    return rows


def validate_events(events, rules, predictions):
    """Check proposed logs against observations, preserving the supplied row order."""
    events = events.copy()
    needed = KEYS + ["event_id", "event", "event_time", "observation_index", "time",
                     "target_index", "target_time", "y", "mlp", "reason",
                     "eligible_for_scoring", "eligible_for_training"]
    require(set(needed) <= set(events), f"Missing event columns: {set(needed) - set(events)}")
    integers(events, KEYS + ["event_id", "observation_index"])
    require(np.array_equal(events.event_id, np.arange(len(events))), "event_id must follow row order from zero")
    for name in ("time", "event_time", "target_time"):
        events[name] = pd.to_datetime(events[name], errors="raise")
    for name in ("eligible_for_scoring", "eligible_for_training"):
        events[name] = flags(events[name])
    events["reason"] = events.reason.fillna("")
    require(np.isfinite(events.mlp).all(), "Nonfinite event prediction")
    expected = rules.set_index(KEYS).to_dict("index")
    scored = predictions.set_index(KEYS).mlp.to_dict()
    pending, seen, finished, clocks = {}, set(), set(), {}
    for row in events.itertuples(index=False):
        key = (row.run, row.bearing, row.input_index)
        require(key in expected, f"Unknown event identity: {key}")
        rule = expected[key]
        clock = (row.event_time, 0 if row.event == "predict" else 1)
        stream = (row.run, row.bearing)
        require(pd.notna(row.event_time) and clock >= clocks.get(stream, (pd.Timestamp.min, 0)),
                "Events must be chronological, with predictions before releases at ties")
        clocks[stream] = clock
        require(row.time == rule["time"], "Prediction time differs from observation")
        if row.event == "predict":
            require(key not in seen, "Duplicate prediction event")
            require(row.event_time == row.time and row.observation_index == row.input_index,
                    "Prediction was not recorded at its input observation")
            require(pd.isna(row.target_time) and pd.isna(row.target_index) and pd.isna(row.y),
                    "Prediction event exposes a future outcome")
            require(not row.eligible_for_scoring and not row.eligible_for_training and not row.reason,
                    "Prediction eligibility must remain false until outcome observation")
            if key in scored:
                require(abs(row.mlp - scored[key]) <= ATOL, "Event prediction differs from scored forecast")
            seen.add(key)
            pending[key] = row.mlp
        else:
            require(key in pending, "Release/discard has no pending prediction in this bearing")
            require(row.event == rule["event"] and row.reason == rule["reason"],
                    "Invalid release/discard: check acquisition gap or forecast horizon")
            require(row.event_time == rule["event_time"] and row.observation_index == rule["observation_index"],
                    "Label/discard time differs from the actual observation time")
            require(abs(row.mlp - pending[key]) <= ATOL, "Forecast changed after prediction")
            for name in ("eligible_for_scoring", "eligible_for_training"):
                require(getattr(row, name) == rule[name], f"Incorrect {name}")
            if row.event == "release":
                require(row.target_time == rule["target_time"] and row.target_index == rule["target_index"],
                        "Released target identity differs from observations")
                require(np.isfinite(row.y) and abs(row.y - rule["y"]) <= ATOL,
                        "Released label differs from this bearing's observation")
            else:
                require(pd.isna(row.target_time) and pd.isna(row.target_index) and pd.isna(row.y),
                        "Discarded pair must not release a label")
            del pending[key]
            finished.add(key)
    require(seen == set(expected) and finished == seen and not pending,
            "Incomplete event log: every causal prediction needs one release or discard")
    return {"status": "passed", "events": len(events), "predictions": len(seen)}


def check_weights(parameters, original):
    require(parameters.keys() == original.keys(), "Frozen parameter names changed")
    require(all(np.array_equal(parameters[k], original[k]) for k in original), "Frozen weights changed")


def verify_runtime(data_dir, checkpoint, predictions):
    """Observe actual model copies before/after every predict call, not metadata claims."""
    source, columns, scale = online_replay.load_checkpoint(checkpoint)
    original = {k: v.copy() for k, v in source.p.items()}
    predict = MLP.predict
    calls = 0

    def watched(net, x):
        nonlocal calls
        check_weights(net.p, original)
        result = predict(net, x)
        check_weights(net.p, original)
        calls += 1
        return result

    raw, segment, _ = load_run(data_dir, 2)
    with patch.object(MLP, "predict", watched), patch.object(Adam, "step", side_effect=ValueError("Training attempted")):
        for bearing in range(1, 5):
            examples = make_examples(raw, segment, 2, bearing)
            inputs = examples[KEYS + ["time", "moving_average"] + columns].copy()
            actual = online_replay.replay_bearing(inputs, source, columns, scale).sort_values(KEYS)
            saved = predictions.loc[predictions.bearing == bearing].sort_values(KEYS)
            require(actual[KEYS].reset_index(drop=True).equals(saved[KEYS].reset_index(drop=True)),
                    "Runtime pair identities differ")
            require(np.allclose(actual.mlp, saved.mlp, rtol=0, atol=ATOL), "Runtime forecasts differ")
    check_weights(source.p, original)
    require(calls == len(predictions), "Expected one frozen predict call per scored pair")
    return {"status": "passed", "prediction_calls": calls,
            "implementation": "code/online_replay.py:replay_bearing"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_synthetic_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "scripts"), pattern="test_check_replay.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    require(result.testsRun > 0 and result.wasSuccessful(), "Synthetic verification tests failed")
    return dict(status="passed", script="scripts/test_check_replay.py", tests_run=result.testsRun,
                failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                source_sha256=digest(ROOT / "scripts/test_check_replay.py"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--events", type=Path, help="Optional proposed-format event log; otherwise marked pending")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/local/results/replay_verification")
    args = parser.parse_args()
    out = args.out.resolve()
    inputs = [args.checkpoint, args.predictions, args.reference_predictions]
    if args.events is not None:
        inputs.append(args.events)
    require(out.is_relative_to((ROOT / "runs").resolve()), "--out must be under this repository's runs/")
    require(all(out != p.resolve().parent for p in inputs), "Keep verification output separate from input files")
    require(all("test3" not in p.name.lower() for p in inputs), "Test 3 files are outside this check")
    synthetic_tests = run_synthetic_tests()
    checkpoint_hash = digest(args.checkpoint)
    rules = pair_rules(read_observations(args.data_dir))
    actual = validate_predictions(pd.read_csv(args.predictions), rules)
    reference = validate_predictions(pd.read_csv(args.reference_predictions), rules)
    max_error = float(np.max(np.abs(actual.mlp.to_numpy() - reference.mlp.to_numpy())))
    require(max_error <= ATOL, "Replay predictions differ from the saved static reference")
    runtime = verify_runtime(args.data_dir, args.checkpoint, actual)
    require(digest(args.checkpoint) == checkpoint_hash, "Checkpoint file changed")
    event_check = {"status": "pending", "reason": "Waiting for delayed-label replay to be completed so we can check it."}
    if args.events is not None:
        event_check = validate_events(pd.read_csv(args.events), rules, actual)
    metrics = metric_rows(actual)
    summary = dict(scope="Frozen Test 2 replay verification; development data",
                   synthetic_tests=synthetic_tests, forecast_checks="passed", runtime_weights=runtime,
                   delayed_label_log=event_check,
                   delayed_label_implementation="Waiting for delayed-label replay to be completed so we can check it.",
                   n=len(actual), prediction_atol=ATOL, reference_max_absolute_error=max_error,
                   checkpoint_sha256=checkpoint_hash, test3_loaded=False,
                   data_sha256={f"test2_{kind}.csv": digest(args.data_dir / f"test2_{kind}.csv")
                                for kind in ("raw", "spectral")},
                   predictions_sha256=digest(args.predictions),
                   reference_predictions_sha256=digest(args.reference_predictions),
                   events_sha256=digest(args.events) if args.events is not None else None,
                   verifier_sha256=digest(Path(__file__)),
                   replay_code_sha256=digest(ROOT / "code/online_replay.py"),
                   python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False, na_rep="unavailable"))
    print(f"PASS: {len(actual)} forecasts; reference max error {max_error:.3g}; runtime weights unchanged")
    if args.events is not None:
        print(f"Delayed-label log: {event_check['status']}")
    print("Waiting for delayed-label replay to be completed so we can check it.")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
