# Timeline and responsibilities

Elissa Ephrem leads data and provenance, Galia Akkl leads models and updates,
and Karim Bteish leads evaluation and diagnostics. All three contribute code,
review, writing, and presentation preparation.

| Week beginning | Milestone | Lead |
|---|---|---|
| 19 October | Data audit, baselines, repository, and proposal due 20 October | All |
| 26 October | Raw-data and timestamp/unit checks; static MLP seed and learning-rate sensitivity | Elissa; Galia |
| 2 November | Causal replay, retraining, and interventions; paired logging and cost profiling | Galia; Karim |
| 9 November | Test-2 tuning and primary protocol freeze; checkpoint-branch pilot and data review | Galia/Karim; Elissa/Karim |
| 16 November | Progress report due 17 November; five-seed development results and selected ablations | All; Galia/Karim |
| 23 November | Frozen Test-3 evaluation, uncertainty, and diagnostic analysis; final scope | Elissa/Karim; all |
| 30 November | Figures, poster, and rehearsal; poster session 4 December | All |
| 7 December | Final report, limitations, and reproducibility; submission 10 December | All |

## Experimental status at proposal submission

Feature-table provenance, exploratory analysis, persistence and ridge baselines,
and a gradient-checked static NumPy MLP have been completed. The baseline metrics
were reproduced on a team member's laptop.

Raw-waveform spot checks, causal online replay, periodic retraining, plasticity
interventions, five-seed comparisons, and checkpoint diagnostics are planned.
The core deliverable is the accuracy-versus-computation comparison. Diagnostic
analysis is secondary to a sound, reproducible core experiment.
