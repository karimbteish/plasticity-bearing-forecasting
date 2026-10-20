# Data directory

`code/fetch_features.py` retrieves six processed IMS feature tables into this
directory and verifies their SHA-256 checksums. Matching existing files are reused;
files with different hashes are preserved and trigger an error.

Downloaded CSVs are excluded from Git. Source references, terms, feature units,
and preprocessing are documented in [data_docs/README.md](../data_docs/README.md).
