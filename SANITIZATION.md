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

`llm_clients.py` sends `thinking: disabled` with DeepSeek requests because the provider
changed the default of deepseek-v4-flash to a thinking mode after the recorded runs; the
recorded runs and judge passes used the non-thinking behaviour.

Records re-generated in the 2026-09-18 revision carry the same fields as the original
records; execution-time bookkeeping fields of the revision run (provider routing,
identity keys, tracebacks) are not distributed.

Retrieved passages in `retrieved_docs` are identified by `doc_id`, `section`, `chunk_id`
and retrieval `score` only. The passage text, which reproduces third-party documents, is
not redistributed; it is regenerated from the knowledge-base index described in
REPRODUCE.md. No reported number uses the passage text.
