# Experiment log

## 20 October 2026

Verified the six processed data tables. After filtering, Test 1 contains
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

Repository audit: frozen Test 2 replay was added on 27 October. It records
predictions chronologically and checks them against batch inference, but does not
release delayed labels or update weights. The 20 October baseline counts and
metrics were reproduced with the pinned dependencies; epoch selection and the
gradient check also matched. Test 3 data were not opened.
The replay reproduced all 3,104 saved Test 2 forecasts within 2.23e-16. Checks of
future-input independence, bearing separation, gap filtering, and empty metrics
passed; the failure-map check used synthetic observations.

Corrected the failure map to include Test 3 B3 and made absent late-life metrics
explicitly unavailable. Development downloads can now select only Tests 1 and 2.

The dated proposal still claims the team's raw-feature extraction script is
ready; no such script is in this repository. Its blanket licence statement also
goes beyond the source evidence; use the qualifications in `data_docs/README.md`.
The proposal's two-branch diagnostic sketch differs from the protocol's three
frozen branches. Reconcile those descriptions before running that study.

Next: agree event ordering, update-clock behaviour at gaps, minimum buffer size,
batch sampling, and Adam-state handling. Then add delayed-label release to the
frozen Test 2 replay and check the event log before enabling ordinary updates.
