# PExpo-Bench publication snapshot

Experimental software and research data supporting "Tool Access Helps a Small Language
Model Approach Flagship Performance on a Personal Exposure Assessment Benchmark".

This snapshot includes the final A0–A4 implementations and the factorial variants used
in the manuscript. The full 1,104-item bank, curated 1,027-item evaluation set, original
response records, scoring inputs and published replication data are retained.

## Contents

- `code/pexpo_bench/`: experiment runner, model clients, retrieval and tools, judges,
  numerical scoring, table statistics and trace diagnostics.
- `data/bank/`: full and curated question banks and the item-level curation record.
- `data/trajectories/main/`: four models × five configurations.
- `data/trajectories/factorial/`: the reported retrieval/rules/budget decomposition.
- `data/trajectories/seeds/`: the reported seed replications.
- `data/scored/`, `data/judges/`, `data/expert_validation/`: recorded scores and validation inputs.
- `data/manifests/`: original run/environment records. Original run identifiers, timestamps,
  errors and retry metadata are retained; private paths and host identifiers are redacted.
- `RESULTS_TABLES.md`: numerical results; `MANIFEST.json`: per-file SHA-256 checksums.

Plotting scripts and rendered figure assets are distributed separately from this code
package. Manuscript figures and their source workbook accompany the journal submission.
This is a publication snapshot; development-only utilities are outside its scope.

See `REPRODUCE.md` for commands and `SANITIZATION.md` for the public redaction policy. Downloading the package is free; new provider API
calls require credentials and may incur charges. Third-party source documents and
the derived retrieval index are not bundled. An external index is required for new
retrieval-based inference; analysis of recorded results does not require it.
