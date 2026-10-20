"""Compare a fresh run to committed reference metrics at printed precision.

This is a reproduction check, not a statistical test of model superiority.
"""
from pathlib import Path
import argparse
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def compare_metrics(reference, actual):
    def read(path):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        mapping = {(row["model"], row["split"]): row for row in rows}
        assert len(mapping) == len(rows), f"Duplicate result rows: {path}"
        return mapping
    expected, observed = read(reference), read(actual)
    assert expected.keys() == observed.keys(), f"Model/split rows differ: {actual}"
    for key, ref in expected.items():
        got = observed[key]
        assert int(ref["n"]) == int(got["n"]), f"Example count changed: {key}"
        for metric in ("mae", "rmse", "late_mae"):
            value = float(got[metric])
            assert math.isfinite(value), f"Non-finite {metric}: {key}"
            assert math.isclose(float(ref[metric]), value, rel_tol=0, abs_tol=1e-6), (
                f"{key} {metric}: reference {ref[metric]}, observed {value}. "
                "The observed run differs from the committed reference tolerance."
            )
    print(f"PASS: {actual.name}, {len(observed)} model/split rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_root.resolve() / "results"
    reference = ROOT / "results"
    if output == reference:
        raise SystemExit("The run directory and committed reference directory must differ.")
    compare_metrics(reference / "baseline_metrics.csv", output / "baseline_metrics.csv")
    compare_metrics(reference / "static_mlp/metrics.csv", output / "static_mlp/metrics.csv")
    baseline = json.loads((output / "run_metadata.json").read_text())
    mlp = json.loads((output / "static_mlp/metadata.json").read_text())
    assert baseline["test3_scored"] is False
    assert mlp["test3_loaded"] is False
    assert baseline["selected_lambda"] == 100
    assert mlp["gradient_check"]["passed"] is True
    assert mlp["gradient_check"]["parameters_checked"] == 49
    print("PASS: selected ridge penalty, gradient check, and recorded evaluation scope")
    print("Reference metrics reproduced within 1e-6 absolute tolerance.")


if __name__ == "__main__":
    main()
