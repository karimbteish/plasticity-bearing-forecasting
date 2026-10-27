"""Replay Test 2 chronologically with a saved, frozen static MLP.

Reuse baseline preprocessing and retrospective forecast eligibility. Prediction
inputs contain no future targets; outcomes are joined only after replay for
scoring. This is an offline replay, not a live stream or an online training run.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from baseline import load_run, make_examples, scores
from static_mlp import MLP


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["run", "bearing", "input_index"]
ATOL = 1e-12


def load_checkpoint(path):
    """Restore inference parameters, feature order, and Test-1 residual scale."""
    with np.load(path, allow_pickle=False) as checkpoint:
        columns = checkpoint["columns"].tolist()
        if len(columns) != 40 or len(set(columns)) != 40:
            raise ValueError("Expected 40 distinct checkpoint feature columns.")
        net = MLP(input_dim=len(columns))
        for key, template in net.p.items():
            value = checkpoint[key].copy()
            if value.shape != template.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint parameter: {key}")
            value.setflags(write=False)
            net.p[key] = value
        target_scale = float(checkpoint["target_scale"])
        if not np.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("Checkpoint target_scale must be finite and positive.")
    # Saved Adam state is deliberately not loaded: there are no updates.
    return net, columns, target_scale


def replay_bearing(inputs, source_model, columns, target_scale):
    """Predict one bearing, one input at a time; accept no outcome table."""
    if not inputs.time.is_monotonic_increasing or not inputs.time.is_unique:
        raise ValueError("Replay inputs must have unique, increasing timestamps.")
    net = deepcopy(source_model)
    for value in net.p.values():
        value.setflags(write=False)
    x = inputs[columns].to_numpy()
    moving_average = inputs.moving_average.to_numpy()
    predictions = inputs[KEYS + ["time"]].copy()
    predictions["mlp"] = [
        moving_average[i] + target_scale * net.predict(x[i:i + 1])[0]
        for i in range(len(inputs))
    ]
    return predictions


def check_reference(predictions, path):
    """Compare the same forecast pairs to an existing static prediction CSV."""
    reference = pd.read_csv(path, parse_dates=["time", "target_time"])
    actual = predictions.sort_values(KEYS).reset_index(drop=True)
    reference = reference.sort_values(KEYS).reset_index(drop=True)
    identity = KEYS + ["target_index", "time", "target_time"]
    pd.testing.assert_frame_equal(actual[identity], reference[identity], check_dtype=False)
    for column in ("y", "mlp"):
        np.testing.assert_allclose(actual[column], reference[column], rtol=0, atol=ATOL)
    return float(np.max(np.abs(actual.mlp.to_numpy() - reference.mlp.to_numpy())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path,
                        help="Optional saved static MLP test2_predictions.csv to verify.")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/local/results/online_replay")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "runs").resolve()):
        parser.error("--out must be under this repository's runs/ directory.")
    source_dirs = {args.checkpoint.resolve().parent}
    if args.reference_predictions is not None:
        source_dirs.add(args.reference_predictions.resolve().parent)
    if out in source_dirs:
        parser.error("Use an output directory separate from the saved static results.")

    source, columns, target_scale = load_checkpoint(args.checkpoint)
    raw, segment, audit = load_run(args.data_dir, 2)
    forecasts, outcomes = [], []
    batch_max_error = 0.0
    for bearing in range(1, 5):
        # This shared helper calibrates ONCE from the first 200 observations and
        # computes causal rolling features. Future values do not fit transforms.
        examples = make_examples(raw, segment, 2, bearing).sort_values("time")
        if set(columns) != {c for c in examples if c.startswith("x_")}:
            raise ValueError("Checkpoint feature columns do not match preprocessing.")
        inputs = examples[KEYS + ["time", "moving_average"] + columns].copy()
        # Target timestamps, values, and retrospective late-life flags stay out
        # of replay_bearing. They are available here only for offline evaluation.
        outcomes.append(examples[KEYS + ["target_index", "target_time", "y", "late"]].copy())
        prediction = replay_bearing(inputs, source, columns, target_scale)
        forecasts.append(prediction)

        batch = inputs.moving_average.to_numpy() + target_scale * source.predict(
            inputs[columns].to_numpy())
        np.testing.assert_allclose(prediction.mlp, batch, rtol=0, atol=ATOL)
        batch_max_error = max(batch_max_error, float(np.max(np.abs(prediction.mlp - batch))))

    # All predictions are fixed before attaching any observed future target.
    predictions = pd.concat(forecasts, ignore_index=True).merge(
        pd.concat(outcomes, ignore_index=True), on=KEYS, validate="one_to_one")
    predictions = predictions[KEYS + ["target_index", "time", "target_time", "mlp", "y", "late"]]
    metrics = dict(model="Frozen MLP replay", split="Validation (Test 2)", n=len(predictions),
                   **scores(predictions.y, predictions.mlp, predictions.late))
    metadata = dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        test3_loaded=False, weights_fixed=True, calibration_fixed=True,
        target_scale=target_scale, data_audit=audit,
        prediction_atol=ATOL, batch_max_absolute_error=batch_max_error,
        reference_predictions=None, reference_max_absolute_error=None,
        evaluation="Future targets joined after replay; eligibility and shutdown filtering are retrospective.",
        numpy=np.__version__, pandas=pd.__version__,
    )
    if args.reference_predictions is not None:
        metadata["reference_max_absolute_error"] = check_reference(predictions, args.reference_predictions)
        metadata["reference_predictions"] = str(args.reference_predictions.resolve())

    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "test2_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Test 2: {len(predictions)} forecasts; validation MAE {metrics['mae']:.12f}")
    print(f"PASS: chronological vs batch predictions (absolute tolerance {ATOL:g})")
    if args.reference_predictions is not None:
        print(f"PASS: saved static predictions; max error {metadata['reference_max_absolute_error']:.3g}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
