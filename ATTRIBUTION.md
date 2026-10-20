# Attribution and AI assistance

## Dataset and feature tables

The IMS Bearing Data Set is credited to J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord
Technical Services (2007), NASA Ames Prognostics Data Repository. The original
collectors were the University of Cincinnati's Center for Intelligent Maintenance
Systems with Rexnord. The official catalog is
https://data.nasa.gov/dataset/ims-bearings.

Processed feature tables originate from `fatemeh-studio/nasa-bearing-anomaly` at
the commit recorded in `data_docs/provenance.json`. The upstream repository's MIT
code license does not independently establish a license for the original dataset.
This repository distributes source references and checksums; data tables and raw
waveforms are retrieved separately. Raw feature extraction has not yet been
independently reproduced.

## Methods and software

The project implements ridge regression, neural-network forward and backward
passes, and Adam using NumPy. The bibliography cites their methodological sources,
NumPy, pandas, and Matplotlib. The planned shrink-and-perturb and ReDo-style
interventions build on the papers cited in the proposal. Their comparative
experiments are future work.

## AI assistance

Claude was used in the initial project draft supplied by the team. ChatGPT assisted
with the revised experimental protocol, literature and data-source checks, code,
LaTeX, and repository documentation. A team member reproduced the reported static
baseline metrics on Windows. The experiment log distinguishes executed results
from planned work and records limitations of the available evidence.
