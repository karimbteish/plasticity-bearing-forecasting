# Experiment log

## 20 October 2026

We verified the six processed data tables. After filtering, Test 1 contains
6,428 training examples and Test 2 contains 3,104 validation examples.

| Model | Validation MAE |
|---|---:|
| Last-value persistence | 0.011278 |
| Moving-average persistence | 0.011243 |
| Ridge regression | 0.011151 |
| Static MLP, seed 42 | 0.011101 |

Test 2 selected ridge lambda 100 and MLP epoch 1. The MLP gradient check
passed; training stopped at epoch 26. Results were reproduced on Windows.

The MLP slightly improved overall MAE, while persistence performed better
late in life. These results do not establish plasticity loss.

Next: implement chronological replay and compare frozen versus continually
updated models.

[Setup and commands](../README.md) · [Detailed results](../results/)

## 14 November 2026

We reran the baselines and the frozen replay added on 27 October. The example
counts, metrics, selected epoch, and gradient check matched the October results.
All 3,104 saved Test 2 forecasts matched within 2.23e-16. Checks of future inputs,
bearing separation, acquisition gaps, and empty metric subsets passed. We used
synthetic observations to check the failure map. Test 3 data were not opened.

Corrected the failure map to include Test 3 B3 and made absent late-life metrics
explicitly unavailable. Development downloads can now select only Tests 1 and 2.

We still need to correct two statements in the proposal: our raw-feature
extraction script is not in the repository, and the licence wording should match
`data_docs/README.md`. We also need to reconcile the proposal's two diagnostic
branches with the three frozen branches in the protocol.

Before the online comparison, we need to agree on event order, update timing at
gaps, minimum buffer size, batch sampling, and Adam-state handling.

### Replay checks (Karim)

We added an independent verifier and 16 synthetic tests. All tests passed.
The frozen replay matched 3,104 static forecasts within 2.22e-16. We checked
weights before and after each prediction; the checkpoint file was unchanged.
MAE was 0.011100621377 and RMSE was 0.025852721617. The 99 late-life pairs belong
to B1. B2–B4 have no late-life subset, so their late MAE is unavailable.

Commands are in the [checklist](replay-checklist.md#run-the-independent-checks).
Results are in [`results/replay_verification`](../results/replay_verification/).

Waiting for delayed-label replay to be completed so we can check it.

## 15 November 2026

The verifier now runs the synthetic tests and forecast checks together and saves
both in the [checks summary](../results/replay_verification/summary.json).
The final pre-push check passed all 16 tests and matched all 3,104 forecasts
within 2.22e-16, with fixed weights. We shortened the checklist and kept the draft
event rules separate from the current frozen replay checks.
