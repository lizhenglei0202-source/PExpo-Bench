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
