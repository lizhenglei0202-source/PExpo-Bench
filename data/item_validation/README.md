# Item validation of the question bank

A stratified random sample of 110 items (seed 20260820, drawn from the 1,027-item
evaluation set, proportional across sub-domain × question type × difficulty;
`sample_manifest.json`) was rated on four binary criteria: clarity, gold correctness,
reference support and difficulty appropriateness (`INSTRUCTIONS.md`). The ratings were
made in two independent LLM passes (OpenAI Codex) from the rating sheet alone (question,
gold answer, unit, difficulty label and the attached gold_references; no model outputs),
with a written reason for every rating of 0, and were checked by PhD-level exposure
scientists.

Files
- `pass1_sheet.csv`, `pass2_sheet.csv`: the completed sheets (item fields as released,
  the four 0/1 criterion columns and the comment).
- `sample_manifest.json`: the sample, its seed, the strata counts and the item ids.
- `INSTRUCTIONS.md`: the instructions issued with the sheets.
- `agreement_report.json`: per-criterion acceptance by both passes, raw agreement and
  Cohen's κ between the passes, acceptance under all four and under three criteria, and
  every rejected item with the reasons given.
- `compute_agreement.py`: recomputes `agreement_report.json` from the two sheets.

Results and interpretation: manuscript Section 3.7 and SI Section C.
