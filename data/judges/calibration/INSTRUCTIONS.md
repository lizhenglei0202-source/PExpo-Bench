# Rater Instructions — Judge Calibration Study (PExpo-Bench)

You are rating answers to open-ended environmental health science questions.
Each row of `human_rating_sheet.csv` contains a question, a reference (gold)
answer, and one anonymized answer. The answers come from several AI systems,
but the system identity is hidden and the row order is randomized — do NOT
try to guess which system produced an answer; rate the text on its merits
only.

## Your task

For each row, read QUESTION, GOLD_ANSWER, and ANSWER_TEXT, then enter an
integer score from 0 to 5 in your column (`rater1_score_0_5` or
`rater2_score_0_5`). Rater 1 and Rater 2 must work independently — do not
discuss rows until both columns are complete.

## Scoring rubric (use exactly this scale)

Score the student's answer against the reference answer on a 0 to 5 scale.

- **5**: Completely correct, covers all key points with accurate detail.
- **4**: Mostly correct, minor omissions or minor imprecision.
- **3**: Partially correct, captures main idea but misses important detail.
- **2**: Weak, shows some understanding but with significant gaps or partial inaccuracies.
- **1**: Mostly incorrect, only tangentially related.
- **0**: Completely wrong, irrelevant, or no substantive answer.

(This is verbatim the rubric used by the automated LLM judge in
`pexpo_bench/runners/run_open_judge.py`, so human and machine scores are
directly comparable.)

## Ground rules

1. Judge against the GOLD_ANSWER, not your own preferred answer. If the
   answer reaches the gold result via a different but valid route, that is
   fine.
2. Numerical answers: treat values within roughly 5% of the gold value (or
   correct after an obvious unit conversion) as correct; wrong order of
   magnitude is a major error.
3. An empty, evasive, or off-topic ANSWER_TEXT scores 0.
4. Extra correct detail beyond the gold answer does not lower the score;
   extra incorrect claims do.
5. Enter whole numbers only (0, 1, 2, 3, 4, 5). Leave no blanks.
6. Do not open `blinding_key.json` or `judge_inputs.jsonl` — they reveal
   which system wrote each answer and would unblind the study.

When both rater columns are complete, return the CSV to the study
coordinator, who will run `analyze_agreement.py`.
