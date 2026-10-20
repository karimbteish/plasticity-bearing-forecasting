# Experiment log

The entries below document the experiments and observations available on
20 October 2026. Computational results and later research plans are distinguished.

## 20 October 2026 — Data and timestamp audit

Six supplied feature tables were matched to a pinned upstream commit; SHA-256
checksums are in `data_docs/provenance.json`. This verifies table provenance but
does not independently verify raw signal extraction.

The corrected loader excludes 200 calibration snapshots per bearing, requires
complete causal history, and rejects pairs spanning irregular acquisition intervals.
It removes two trailing shutdown rows from Test 2. The resulting development data
contain 6,428 training pairs and 3,104 validation pairs. Test 1 has 24 gaps longer
than 15 minutes, including a 148.43-hour gap; Test 2 has none. Counts and plots are
in `results/data_audit.csv` and `paper/figures/data_overview.png`.

## 20 October 2026 — Persistence and ridge baselines

Command run on the team member's Windows laptop:

```powershell
.\.venv\Scripts\python.exe code\baseline.py --data-dir data
```

Test 2 selected ridge lambda 100. Validation MAEs were 0.011278 for last-value
persistence, 0.011243 for moving-average persistence, and 0.011151 for ridge.
Last-value persistence had the lowest late-life MAE, 0.069358. The reported
baseline computation time was 0.958 seconds on that laptop. Detailed reference
scores and the penalty sweep are committed in `results/`.

Interpretation: persistence is a strong baseline. A small aggregate MAE change
does not establish useful improvement during degradation. Scores are selected
validation performance, not independent final-test performance.

## 20 October 2026 — Static neural baseline

Command run on the same laptop:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
.\.venv\Scripts\python.exe code\static_mlp.py --data-dir data
```

The finite-difference check passed for all 49 parameters of a small network,
with maximum absolute error approximately 2.08e-10. The 40–64–64–1 network used
seed 42 and 6,849 parameters. It selected epoch 1 and stopped at epoch 26.
Training MAE was 0.003361; validation MAE 0.011101, RMSE 0.025853, and late MAE
0.076161. Reported computation time was 1.36 seconds, excluding plotting.

The laptop's terminal results are recorded in
`results/static_mlp/windows_reproduction.json`; the separate `metadata.json`
records the development-environment reference run. Runtime and environment fields
are intentionally kept distinct. The saved checkpoint was reloaded in the
development environment and reproduced the stored validation predictions within
1e-12 absolute tolerance.

Interpretation: further epochs reduce training error but worsen validation MAE.
This is a generalization finding. Static evaluation does not measure the ability
to learn from a changing stream and does not establish loss of plasticity.

## Planned experiments

Raw-waveform spot checks, causal online replay, periodic retraining, the two
plasticity interventions, multiple seeds, and checkpoint diagnostic comparisons.
No results for these planned experiments are claimed in this repository.
