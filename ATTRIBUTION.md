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

We implement ridge regression, the MLP, and Adam using NumPy. The bibliography
cites the methods and the NumPy, pandas, and Matplotlib libraries. Our planned
shrink-and-perturb and ReDo-style experiments build on the papers in the proposal.

## AI assistance

We used Claude for the initial project draft and ChatGPT for help with the
protocol, literature and data-source checks, code, LaTeX, and documentation.
ChatGPT also helped build and run the replay verification tests. A team member
reproduced the static baseline metrics on Windows. We record completed checks
and remaining work in the experiment log.
