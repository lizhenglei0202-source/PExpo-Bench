# Item validation instructions / 题目验证评分说明

**What this is / 这是什么.** A stratified random sample of 110 PExpo-Bench questions (seed 20260820, drawn from the 1,027-item evaluation set, proportional across sub-domain × question type × difficulty; `sample_manifest.json`), shown in the version of the released question bank. Each rating pass rates every question independently.
从评测集中按子领域 × 题型 × 难度成比例分层随机抽取的 110 道题目（种子 20260820），按公开版题库的题目版本呈现。每次评分各自独立进行。

**Files / 文件.** The two rating sheets are identical. Each pass works ONLY in its own sheet, without any exchange until both are complete. 两份评分表内容相同；每次评分只填写自己的表，全部完成前不交流。

**Per row, fill the four criterion columns with 1 (satisfactory / 合格) or 0 (not satisfactory / 不合格):**

| Column | Criterion | 标准 |
|---|---|---|
| `clarity_0_1` | The question unambiguously asks for a specific quantity or judgment | 题目无歧义地询问某个具体量或判断 |
| `gold_correctness_0_1` | The gold answer is factually correct under the cited source | 金标答案在所引来源下事实正确 |
| `reference_support_0_1` | The attached gold references (see `gold_references` column) are sufficient to justify the gold answer | 附带的金标参考足以支撑金标答案 |
| `difficulty_appropriateness_0_1` | The labelled difficulty matches your judgment | 标注难度与您的判断一致 |

**Rules / 规则.**
- Enter only 0 or 1, no blanks, no 0.5. 只填 0 或 1，不留空，不填 0.5。
- If any criterion gets 0, write a short reason in `comment` (English or Chinese). 任何一项为 0 时请在 comment 列写明原因。
- Judge the question as printed, with the gold answer and references shown in the sheet. Do not consult model outputs or any earlier rating. 仅依据表中的题目、金标答案和附带参考评判；不要参考模型输出或以往的任何评分。
- The `gold_references` column gives the full quoted passages (no truncation). Items whose qid starts with `TS` are programmatic items; their reference is a table locator (for example "EPA EFH 2011 Table 6-1") rather than a quoted passage. `gold_references` 列给出完整引文段落（不截断）；qid 以 TS 开头的程序化题只给出表格出处而非引文段落。
- An item is ACCEPTED only if both passes give 1 on all four criteria. 一道题只有在两次评分四项都给 1 时才算通过。

**When both sheets are complete / 两份都完成后**, `compute_agreement.py` computes the acceptance rate, per-criterion agreement between the passes (Cohen's κ) and the rejected-item list with the comments.

These instructions were issued with the sheets. The sheets were completed in two independent LLM passes (OpenAI Codex) and checked by PhD-level exposure scientists (`README.md`).
