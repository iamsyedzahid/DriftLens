# Data

`python scripts/downloadData.py` fetches the UCI Bank Marketing archive
(https://archive.ics.uci.edu/dataset/222/bank+marketing, CC BY 4.0) and extracts
`bank-additional-full.csv` into `data/raw/`. It also writes `data/raw/SOURCE.json`, which records
the URL, retrieval date (UTC) and SHA-256 checksum. Raw data is not committed.

Cite: S. Moro, P. Cortez, P. Rita, "Bank Marketing [Dataset]", UCI ML Repository, 2012, doi:10.24432/C5K306.
