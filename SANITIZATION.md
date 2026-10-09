# Public snapshot redaction

Private filesystem locations, workstation identifiers, credentials and private
service endpoints are omitted or replaced by explicit REDACTED markers.
Package-relative paths and public scientific/resource URLs remain usable.
Original run identifiers, seeds, failure states, retry counts, question IDs,
model responses, tool traces and numerical scores are retained. Redaction does
not remove experimental runs. Original records are retained privately; manifest
checksums describe these public files and may differ from private source hashes.

The blank .env.example contains configuration names only. Supply credentials and
any external retrieval index locally. Do not commit a populated .env file.

Response records carry the fields listed in `RECORD_KEYS` of the runner; execution-time
bookkeeping (provider routing, identity keys, tracebacks) is not distributed. Retrieved
passages in `retrieved_docs` are identified by `doc_id`, `section`, `chunk_id` and
retrieval `score` only. The passage text, which reproduces third-party documents, is
not redistributed; it is regenerated from the knowledge-base index described in
REPRODUCE.md. No reported number uses the passage text.

The bank files contain the final question set only: question, answer, unit, tolerance
and scoring specification, rationale, gold references, source tags and physical
constraints. Internal editing metadata is not distributed.

`data/judges/grounding_judgments.jsonl` holds one judgment row for every main-grid
response (1,027 items × 20 cells = 20,540 rows). All rows were produced in a single judge
pass with the released judge code; a row whose extractor or entailment call failed was
re-run until no failed call remained, so every row is a completed judgment.
