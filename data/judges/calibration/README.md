# Judge calibration

The calibration study contains 400 answers: 100 questions sampled proportionally
across subdomain and difficulty, with the A3 response from each of the four
evaluated models. Sampling and row shuffling use seed 42.

Two model judges score every answer with the same 0–5 rubric. Two human raters
independently score the blinded rows. Model identity is excluded from the rating
sheet; the mapping in `blinding_key.json` is used after rating is complete.

## Files

| File | Contents |
|---|---|
| `sample_manifest.json` | Sampled questions, strata, allocation and seed |
| `human_rating_sheet.csv` | Completed anonymous ratings for 400 answers |
| `INSTRUCTIONS.md` | Rating instructions and rubric |
| `blinding_key.json` | Sample-code to question/model mapping |
| `judge_inputs.jsonl` | Inputs for the two model judges |
| `per_row_double_judge.jsonl` | Recorded scores from both judges |
| `agreement_report.json` | Agreement statistics |
| `run_double_judge.py` | Execute model judging with local credentials |
| `analyze_agreement.py` | Recompute agreement without API calls |

The English and Chinese instruction documents provide the same rating protocol
in accessible formats.

## Analysis

From the package root:

```bash
python data/judges/calibration/analyze_agreement.py --out analysis_outputs/judge_agreement.json
```

Outputs include judge means, Pearson/Spearman correlations, exact and within-one
agreement, weighted Cohen's kappa, rater agreement, and comparisons against the
mean human score. Questions, reference answers and responses in the judge inputs
follow the recorded 2,500-character truncation policy.
