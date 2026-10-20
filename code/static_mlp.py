"""A static, from-scratch NumPy MLP for one-hour vibration forecasting.

Command-line interface:
    python code/static_mlp.py --data-dir data

Requires baseline.py in the same code folder and the existing requirements.txt.
Training: Test 1. Checkpoint selection: Test 2. Test 3 is never loaded.
The model predicts a correction to moving-average log-RMS, like the ridge model.
Forward propagation, backpropagation and Adam are implemented here with NumPy.

References already in paper/refs.bib: Kingma & Ba (2015), Adam, arXiv:1412.6980;
Harris et al. (2020), NumPy, doi:10.1038/s41586-020-2649-2. Dataset provenance
and transformations are documented in baseline.py and data_docs/README.md.
Prepared with ChatGPT assistance; attribution is recorded in ATTRIBUTION.md.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline import load_run, make_examples, scores


class MLP:
    """Two ReLU hidden layers and a linear scalar output (float64)."""

    def __init__(self, input_dim=40, hidden=(64, 64), seed=42):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden
        self.p = {
            "W1": rng.normal(0, np.sqrt(2 / input_dim), (input_dim, h1)),
            "b1": np.zeros((1, h1)),
            "W2": rng.normal(0, np.sqrt(2 / h1), (h1, h2)),
            "b2": np.zeros((1, h2)),
            # Small output weights give near-zero corrections and nonzero hidden-layer gradients.
            "W3": rng.normal(0, 0.01 / np.sqrt(h2), (h2, 1)),
            "b3": np.zeros((1, 1)),
        }

    def forward(self, x):
        z1 = x @ self.p["W1"] + self.p["b1"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.p["W2"] + self.p["b2"]
        a2 = np.maximum(z2, 0)
        out = a2 @ self.p["W3"] + self.p["b3"]
        return out, (x, z1, a1, z2, a2)

    def loss_and_gradients(self, x, y):
        pred, (x, z1, a1, z2, a2) = self.forward(x)
        error = pred - y
        loss = float(np.mean(error ** 2))
        d3 = 2 * error / len(x)
        d2 = (d3 @ self.p["W3"].T) * (z2 > 0)
        d1 = (d2 @ self.p["W2"].T) * (z1 > 0)
        grads = {
            "W1": x.T @ d1, "b1": d1.sum(axis=0, keepdims=True),
            "W2": a1.T @ d2, "b2": d2.sum(axis=0, keepdims=True),
            "W3": a2.T @ d3, "b3": d3.sum(axis=0, keepdims=True),
        }
        return loss, grads

    def predict(self, x):
        return self.forward(x)[0].ravel()


class Adam:
    """Adam with bias correction; no weight decay or gradient clipping."""

    def __init__(self, params, lr=1e-3):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def check_gradients():
    """Central finite differences on ALL 49 parameters of a small version."""
    rng = np.random.default_rng(112)
    net = MLP(input_dim=3, hidden=(5, 4), seed=7)
    x, y = rng.normal(size=(9, 3)), rng.normal(size=(9, 1))
    _, analytical = net.loss_and_gradients(x, y)
    abs_errors, rel_errors = [], []
    epsilon = 1e-6
    for key, param in net.p.items():
        for index in np.ndindex(param.shape):
            old = param[index]
            param[index] = old + epsilon
            lp = np.mean((net.forward(x)[0] - y) ** 2)
            param[index] = old - epsilon
            lm = np.mean((net.forward(x)[0] - y) ** 2)
            param[index] = old
            numerical = (lp - lm) / (2 * epsilon)
            actual = analytical[key][index]
            absolute = float(abs(numerical - actual))
            relative = absolute / max(1e-8, abs(numerical) + abs(actual))
            abs_errors.append(absolute)
            rel_errors.append(relative)
            if absolute > 1e-7 and relative > 1e-5:
                raise AssertionError(f"Gradient check failed: {key}{index}")
    result = {"parameters_checked": len(abs_errors), "max_absolute_error": max(abs_errors),
              "max_relative_error": max(rel_errors), "passed": True}
    print(f"Gradient check PASS: {len(abs_errors)} parameters; "
          f"max absolute error {max(abs_errors):.2e}", flush=True)
    return result


def data_for_run(data_dir, run):
    raw, segment, audit = load_run(data_dir, run)
    ex = pd.concat([make_examples(raw, segment, run, b) for b in range(1, 5)], ignore_index=True)
    return ex, audit


def plot_outputs(history, predictions, out, best_epoch):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric in zip(axes, ("rmse", "mae")):
        ax.plot(history.epoch, history[f"train_{metric}"], label="Train: Test 1", color="#245a81")
        ax.plot(history.epoch, history[f"validation_{metric}"], label="Validation: Test 2", color="#bb6b27")
        ax.axvline(best_epoch, color="#666666", ls=":", label="Selected epoch")
        ax.set(xlabel="Training epoch", ylabel=f"{metric.upper()} (log10 RMS units)")
        ax.set_title("Checkpoint selection uses validation MAE" if metric == "mae" else "Training progress")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=180)
    plt.close(fig)

    va = predictions[2]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    start = pd.to_datetime(va.time).min()
    for b, ax in enumerate(axes.flat, 1):
        group = va.loc[va.bearing == b].copy()
        times = pd.to_datetime(group.target_time)
        days = (times - start).dt.total_seconds() / 86400
        # Missing or filtered intervals separate the plotted prediction segments.
        gap = times.diff().dt.total_seconds().gt(660)
        for column, label, color, style in (
            ("y", "Observed", "#222222", "-"),
            ("persistence", "Last-value persistence", "#bb6b27", "--"),
            ("mlp", "Static MLP", "#245a81", "-"),
        ):
            values = group[column].copy()
            values.loc[gap] = np.nan
            ax.plot(days, values, label=label, color=color, ls=style, lw=1)
        ax.set_title(f"Test 2 - bearing B{b}" + (" (documented failure)" if b == 1 else ""))
        ax.set(xlabel="Days since first eligible validation input", ylabel="Target log10(RMS / 1 g)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("One-hour forecasts from the selected static MLP; target-time alignment", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "validation_forecasts.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results/static_mlp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.lr <= 0:
        ap.error("Epochs, patience, batch size, and learning rate must be positive.")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    gradient_check = check_gradients()
    tr, train_audit = data_for_run(args.data_dir, 1)
    va, validation_audit = data_for_run(args.data_dir, 2)
    columns = [c for c in tr.columns if c.startswith("x_")]
    X, V = tr[columns].to_numpy(), va[columns].to_numpy()
    residual = (tr.y - tr.moving_average).to_numpy()
    # Training-only scaling improves optimization; forecasts return to log-RMS units.
    target_scale = max(float(residual.std()), 1e-8)
    target = (residual / target_scale).reshape(-1, 1)
    net = MLP(input_dim=X.shape[1], seed=args.seed)
    optimizer = Adam(net.p, lr=args.lr)
    batch_rng = np.random.default_rng(args.seed + 1)
    count = sum(v.size for v in net.p.values())
    print(f"Train: {len(tr)} examples; validation: {len(va)}; inputs: {X.shape[1]}", flush=True)
    print(f"Static MLP 40-64-64-1: {count} parameters; seed={args.seed}", flush=True)
    print("Test 2 selects the checkpoint; Test 3 is not loaded.", flush=True)

    def measure(epoch):
        train_pred = tr.moving_average.to_numpy() + target_scale * net.predict(X)
        val_pred = va.moving_average.to_numpy() + target_scale * net.predict(V)
        ts = scores(tr.y, train_pred, tr.late)
        vs = scores(va.y, val_pred, va.late)
        row = dict(epoch=epoch, updates=optimizer.t,
                   **{f"train_{k}": v for k, v in ts.items()},
                   **{f"validation_{k}": v for k, v in vs.items()})
        if not all(np.isfinite(value) for value in row.values()):
            raise FloatingPointError("Nonfinite metric; inspect optimization before continuing.")
        return row

    rows = [measure(0)]
    best_mae, best_epoch, best_params, best_state = np.inf, 0, None, None
    for epoch in range(1, args.epochs + 1):
        order = batch_rng.permutation(len(X))
        for offset in range(0, len(X), args.batch_size):
            batch = order[offset:offset + args.batch_size]
            loss, grads = net.loss_and_gradients(X[batch], target[batch])
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            optimizer.step(net.p, grads)
        row = measure(epoch)
        rows.append(row)
        if row["validation_mae"] < best_mae:
            best_mae, best_epoch = row["validation_mae"], epoch
            best_params = {k: v.copy() for k, v in net.p.items()}
            best_state = {"step": optimizer.t, **{f"m_{k}": v.copy() for k, v in optimizer.m.items()},
                          **{f"v_{k}": v.copy() for k, v in optimizer.v.items()}}
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train MAE {row['train_mae']:.6f}; "
                  f"validation MAE {row['validation_mae']:.6f}; best epoch {best_epoch}", flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break
    net.p = best_params
    history = pd.DataFrame(rows)
    history.to_csv(args.out / "training_history.csv", index=False)

    # Fixed lambda=100 was selected by the preceding baseline script on Test 2.
    design = np.c_[X, np.ones(len(X))]
    penalty = 100 * np.eye(design.shape[1])
    penalty[-1, -1] = 0
    ridge_w = np.linalg.solve(design.T @ design + penalty, design.T @ residual)
    metrics, per_bearing, predictions = [], [], {}
    for run, ex, inputs in ((1, tr, X), (2, va, V)):
        preds = {
            "Last-value persistence": ex.persistence.to_numpy(),
            "Moving-average persistence": ex.moving_average.to_numpy(),
            "Ridge residual": ex.moving_average.to_numpy() + np.c_[inputs, np.ones(len(ex))] @ ridge_w,
            "Static MLP": ex.moving_average.to_numpy() + target_scale * net.predict(inputs),
        }
        split = "Train (Test 1)" if run == 1 else "Validation (Test 2)"
        for name, pred in preds.items():
            metrics.append(dict(model=name, split=split, n=len(ex), **scores(ex.y, pred, ex.late)))
            for bearing in range(1, 5):
                mask = ex.bearing.to_numpy() == bearing
                errors = pred[mask] - ex.y.to_numpy()[mask]
                per_bearing.append(dict(model=name, split=split, bearing=bearing, n=int(mask.sum()),
                                        mae=float(abs(errors).mean()), rmse=float(np.sqrt((errors ** 2).mean()))))
        pred_frame = ex[["run", "bearing", "input_index", "target_index", "time", "target_time",
                         "y", "persistence", "moving_average", "late"]].copy()
        pred_frame["ridge"] = preds["Ridge residual"]
        pred_frame["mlp"] = preds["Static MLP"]
        pred_frame.to_csv(args.out / f"test{run}_predictions.csv", index=False)
        predictions[run] = pred_frame
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(per_bearing).to_csv(args.out / "per_bearing_metrics.csv", index=False)
    np.savez(args.out / "model.npz", **net.p, **best_state, target_scale=target_scale,
             columns=np.array(columns), best_epoch=best_epoch, seed=args.seed)
    computation_seconds = time.perf_counter() - start
    metadata = dict(seed=args.seed, architecture=[40, 64, 64, 1], parameters=count,
                    learning_rate=args.lr, batch_size=args.batch_size, max_epochs=args.epochs,
                    patience=args.patience, selected_epoch=best_epoch, stopped_epoch=epoch,
                    selected_checkpoint_updates=int(best_state["step"]), total_updates=optimizer.t,
                    checkpoint_selection="Lowest Test-2 validation MAE among trained epochs; no Test-2 gradients",
                    target="1-hour log10 RMS residual relative to trailing mean log-RMS",
                    target_scale_train_only=target_scale, test3_loaded=False,
                    validation_is_used_for_selection=True, gradient_check=gradient_check,
                    computation_seconds=computation_seconds, python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                    data_audits=[train_audit, validation_audit],
                    data_sha256={f"test{r}_{kind}.csv": hashlib.sha256(
                        (args.data_dir / f"test{r}_{kind}.csv").read_bytes()).hexdigest()
                        for r in (1, 2) for kind in ("raw", "spectral")})
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_outputs(history, predictions, args.out, best_epoch)
    print(f"\nSelected epoch: {best_epoch}; computation seconds: {computation_seconds:.2f}")
    print(metrics.round(6).to_string(index=False))
    print(f"\nOutputs: {args.out.resolve()}")
    print("This is one static-model seed, not a plasticity experiment or final test result.")


if __name__ == "__main__":
    main()
