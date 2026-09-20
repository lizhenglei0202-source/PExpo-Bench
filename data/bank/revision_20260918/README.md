# Gold-standard revision of 18 September 2026

All 1,104 questions were re-examined after the expert validation (SI Section C) had
questioned gold answers: two independent model-based screening passes (GPT-5.4 and
DeepSeek-V4, each covering every item) plus 194 manual arithmetic or source checks
flagged 638 evaluation-set items, of which 135 had a confirmed defect in the answer,
unit, stated conditions or explanation. The authors then reviewed all 1,104 items one by
one, adjudicating the flagged items against the primary sources.

| action | items | meaning |
|---|---:|---|
| rerun | 97 | question text changed; every response in every experiment re-generated |
| rescore | 52 | true/false or calculation answer/unit corrected; original responses re-scored |
| rejudge | 35 | open-ended answer corrected; original responses re-judged with the original protocol |
| metadata_only | 57 | rationale or unit wording only; scores unchanged |
| retain | 397 | kept (239 of them with supplemented source citations) |
| historical_excluded | 48 | items outside the 1,027-item evaluation set; remain excluded |

Files

- `revision_manifest.json`: one entry per reviewed item with the review group, the
  author decision, the fields changed, adjudication notes (English translation and
  the original Chinese), the source used, and the item before and after.
- `evidence_ledger.json`: the primary-source evidence consulted per item.
- `post_freeze_evidence_addendum.json`: a citation located after the generation freeze
  (no question or answer changed).
- `final_assembly_summary.json`: counts of re-generated, re-scored and re-judged rows.

Re-generated responses (5,468: 1,940 main grid, 1,440 factorial, 2,088 seed
replication) used the recorded prompts, tool schemas, decoding settings, budgets
and retrieval index; the returned model snapshot identifiers were identical to the
original runs. Calculation items are scored with `analysis/numeric_revision.py`,
which keeps full-precision reference values and accepts the displayed rounding; the
same rule is applied to every calculation item. The reference-grounding judgments
(`data/judges/grounding_judgments.jsonl`) were re-run for the 16 subsample items whose
question text changed (320 rows; same extractor, classifier models and retrieval index),
and kept for all other rows.
The seed-replication files contain the 545 true/false and calculation items used in
the analysis; open-ended seed responses were not re-generated and are not
distributed.
