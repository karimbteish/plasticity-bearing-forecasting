# Collaboration and reproducibility

Elissa Ephrem leads data provenance and feature verification. Galia Akkl leads
model implementation and update strategies. Karim Bteish leads evaluation,
diagnostics, and computational accounting. Writing and code review are shared.

Changes are organized through topic branches and pull requests. Commits record
actual contributions under each contributor's Git identity; co-authorship denotes
joint work. Experiment records identify the configuration, source revision,
random seed, dataset split, command, result, and interpretation.

Pipeline changes are checked against the committed static reference metrics.
Changes to expected values are accompanied by a documented methodological reason.
Downloaded data, virtual environments, checkpoints, and per-example prediction
files are regenerated locally and excluded from version control.

Test 2 informs development choices. The primary protocol is frozen before Test-3
model scores are evaluated. Chronological adaptation uses only targets available
at each prediction time. Negative findings and failed experiments remain part of
the research record when they inform subsequent decisions.
