# Expert Validation — Rater Instructions / 专家验证评分说明

**What this is.** A stratified random sample of 110 PExpo-Bench questions (seed 20260820, drawn from the curated 1,027-item evaluation set, proportional across sub-domain × question type × difficulty; see `sample_manifest.json`). Two PhD-level exposure scientists each rate every question independently.

**Files.** `rater1_sheet.csv` and `rater2_sheet.csv` are identical. Each rater works ONLY in their own file, without discussing any item until both are complete. 两位评审各自只填写自己的文件，全部完成前不讨论任何题目。

**Per row, fill the four criterion columns with 1 (satisfactory / 合格) or 0 (not satisfactory / 不合格):**

| Column | Criterion | 标准 |
|---|---|---|
| `clarity_0_1` | The question unambiguously asks for a specific quantity or judgment | 题目无歧义地询问某个具体量或判断 |
| `gold_correctness_0_1` | The gold answer is factually correct under the cited source | 金标答案在所引来源下事实正确 |
| `reference_support_0_1` | The attached gold references (see `gold_references` column) are sufficient to justify the gold answer | 附带的金标参考足以支撑金标答案 |
| `difficulty_appropriateness_0_1` | The labelled difficulty matches your judgment | 标注难度与您的判断一致 |

**Rules.**
- Enter only 0 or 1 — no blanks, no 0.5. 只填 0 或 1，不留空。
- If any criterion gets 0, write a short reason in `comment` (English or Chinese). 任何一项为 0 时请在 comment 列写明原因。
- Judge the question as printed; do not consult model outputs (there are none in this sheet). 仅评题目本身。
- An item is ACCEPTED only if both raters give 1 on all four criteria (4-of-4 × 2).

**When both sheets are complete**, hand them back; the analysis script will compute the acceptance rate, per-criterion inter-rater agreement (Cohen's κ), and the rejected-item list with your comments as the resolution log.
