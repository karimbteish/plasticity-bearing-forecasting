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