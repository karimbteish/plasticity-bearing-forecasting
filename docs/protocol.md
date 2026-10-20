# Experimental protocol

## Objective and scope

We will test whether shrink-and-perturb and selective neuron resets improve
one-hour-ahead bearing vibration forecasts compared with ordinary continual
training and periodic retraining. The target is log10(RMS / 1 g). All neural
comparisons use the same 40–64–64–1 MLP architecture.

The static baselines have been run. The online comparison and diagnostic study
below are planned. The [proposal](../paper/main.pdf) contains the wider study plan
and timeline.

## Data and split

- Test 1 trains the source models.
- Test 2 selects hyperparameters and update settings.
- Test 3 is evaluated after those choices are fixed. Its descriptive statistics
  have already been inspected; its model scores have not been used for selection.

Each bearing starts from a separate copy of the source model. Preprocessing and
forecast eligibility follow the [data documentation](../data_docs/README.md).
Predictions are recorded before their targets are used for learning. A training
pair becomes available only when its one-hour-ahead target has been observed.

## Models and update rules

Persistence, ridge regression, and the frozen MLP provide reference forecasts.
The adaptive comparison consists of:

| Strategy | Update rule |
|---|---|
| Ordinary updating | Continue training the existing network on recent examples. |
| Periodic retraining | Reinitialize the network and optimizer periodically, then train on the same recent examples. Continue updating between restarts. |
| Shrink-and-perturb | Slightly shrink the existing weights and add random initialization noise before further training. |
| Selective resets | Reinitialize low-activation units, zero their outgoing weights, and clear the affected optimizer state. |

The initial plan updates every six hours using at most the latest 200 eligible
pairs whose targets have arrived. Buffers are cleared at acquisition gaps.
All strategies receive the same examples and are compared separately at fixed
budgets of 25 and 100 gradient steps per update. Test 2 selects intervention
strength and frequency, starting with 24- and 48-hour intervals. Actual runtime
and reset overhead are recorded alongside training steps.

## Reporting

We will use five paired seeds and report per-bearing and overall MAE/RMSE,
late-life error, mean underprediction, and computation costs. Late-life targets
fall in the final 10% of elapsed live-run time for documented failed bearings.
Per-seed differences will accompany aggregate results. The four bearings in a
run share a rig and are not independent experimental replications.

## Secondary diagnostic study

At selected checkpoints, copies of the ordinary model will receive continued
training, an intervention, or fresh retraining with the same data and step budget.
They will then remain frozen over the following 24-hour forecast window.

Dormant-unit fraction and effective rank will be measured before these branches
are trained. We will compare their ability to predict subsequent intervention
benefit with recent forecast error and accumulated update count. Diagnostic
choices use Test 2 and are fixed before evaluation on Test 3. The forecasting
comparison remains the primary deliverable if the diagnostic evidence is weak.
