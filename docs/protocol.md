# Experimental protocol

## Objective and scope

The study compares how a small neural network maintains one-hour-ahead vibration
forecasts as bearing measurements arrive. The primary comparison is ordinary
continual updating versus periodic fresh retraining, shrink-and-perturb, and
ReDo-style selective resetting under comparable computational budgets. The
secondary comparison tests whether internal diagnostics predict intervention
benefit beyond recent forecast error and accumulated training effort.

The task is regression on log10(RMS / 1 g), using one 40–64–64–1 architecture.
Persistence and ridge provide simple reference models. The study does not assume
that plasticity loss must occur or that a reset must improve forecasting.

## Data and split

Test 1 trains source models, Test 2 selects settings, and Test 3 supplies evaluation
after the primary protocol is frozen. Earlier descriptive inspection of Test 3
is acknowledged; its model scores do not enter the proposal results or development
selection. Each bearing begins from an independent copy of the same selected
source model. Separate bearing histories are not concatenated into an artificial
sequence of new machines.

The static pipeline is specified in `data_docs/README.md`. Input and target
timestamps are retained so online replay can enforce label availability. A target
is eligible for an update only when its target timestamp is no later than the
current replay time. Predictions are recorded before the associated labels are
used for learning.

## Online updates and computational budget

Updates occur every six hours and use the latest 200 eligible matured examples.
The buffer is cleared at acquisition gaps. An update requires at least 32 matured
examples; otherwise it is skipped and forecasts remain in the prediction record.
Minibatches contain 64 examples sampled with replacement, using the same sampling
rule across methods.

Ordinary updating is tuned first on Test 2 using learning rates 1e-4, 3e-4, and
1e-3. The selected learning rate is shared across the primary comparisons.
Update allowances are 25 or 100 gradient steps. Restart and intervention
frequencies are 24 or 48 hours. Each intervention has at most eight combinations
of step allowance, frequency, and strength.

The proposed shrink/noise pairs are (0.99, 0.001) and (0.95, 0.01), with noise
drawn according to the network's initialization distribution. ReDo-style relative
activation thresholds are 0.01 and 0.05. These values define a development grid;
they are not claimed to be optimal settings from prior literature.

The adaptive trajectories start with the same source weights and freshly
initialized Adam state. Ordinary updating retains its optimizer state. Full
retraining resets all weights and optimizer state, then continues updating between
restarts. ReDo-style resets reinitialize incoming weights, zero outgoing weights,
and clear affected optimizer moments, including those for biases. The primary
shrink-and-perturb condition retains optimizer moments; an optimizer-reset-only
control and a reset-moments sensitivity separate those effects where needed.

Initialization, minibatch schedules, and intervention randomness use separate
random generators. Computation records include gradient updates, resets,
diagnostic overhead, initialization, and elapsed wall time. A retraining
sensitivity with four times the update allowance is reported on an accuracy-cost
curve separately from equal-step comparisons.

## Evaluation

Five paired seeds share source initializations and minibatch schedules where
applicable. Reporting includes pooled and macro-average per-bearing MAE, RMSE,
late-life MAE, mean underprediction, runtime, update counts, and reset counts.
Late-life targets fall in the final 10% of elapsed live-run time for documented
failed bearings; this retrospective subset does not enter model inputs.

Per-bearing and per-seed results accompany paired method differences. Descriptive
95% moving-block bootstrap intervals use 24-hour blocks with a 48-hour sensitivity,
keeping co-mounted bearings grouped. These intervals describe the recorded runs
rather than population-level industrial reliability. Ablations vary buffer size,
update interval, and range compression one factor at a time.

## Diagnostic branch experiment

Every eligible 24-hour checkpoint produces copies of the ordinary continually
trained model and its matured buffer. The branches receive ordinary continuation,
the intervention selected on Test 2, or fresh initialization. Each branch uses the
same data and step allowance, then remains frozen while forecasting the next
non-overlapping 24-hour interval. These branches are separate from the main online
trajectories.

The intervention gain is ordinary-branch MAE minus intervention-branch MAE.
Positive gain indicates a forecasting benefit. Diagnostics measured before
branching are dormant-unit fraction and centered effective rank divided by hidden
width, with zero effective rank for a zero activation matrix. Recent matured-label
error and cumulative gradient updates are comparator signals.

Directions and thresholds are selected on Test 2 and evaluated on Test 3 using
Spearman correlation and AUROC for positive gain. AUROC is undefined when only one
outcome class occurs. Correlated checkpoints are not treated as independent
machines or independent experimental replications.

## Interpretation and implementation checks

A synthetic changing-target regression stream checks gradient and reset behavior.
A separate probe on copied models compares warm and fresh learning curves using
matched output initialization, reset optimizer state, identical random targets,
and equal training budgets. Initial losses accompany final losses, so output-scale
differences remain visible. Probe tasks do not modify the deployed forecaster.

Better probe trainability without better forecasts would indicate that the probe
captures a property that is not the limiting factor in this application.
Persistence or fresh retraining may outperform the interventions. Little observed
plasticity loss is a valid result. Limited physical runs and correlated bearing
histories constrain generalization regardless of the number of seeds or windows.
