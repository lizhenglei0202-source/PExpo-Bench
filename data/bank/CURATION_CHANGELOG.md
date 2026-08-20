# PExpo-Bench bank changelog — 2026-08-11 patch

Written as PATCHED COPIES (`*.patched_20260811.yaml`); the original bank files are
untouched. Every new gold was verified by recomputing it from parameters stated in
the question text (`analysis/bank_patch_20260811.py`). Retirement is a soft delete
via `_retired_20260811`; retired items remain in the files and raw run logs but are
excluded from curated scoring. Effective item count: 1104 → 1027 (one retired item appears in both the defect and key-balance lists).

## Gold-answer fixes (10 — all were scored ~0 in every cell, so scores only increase)

| qid | new gold | reason |
|---|---|---|
| S3-0184 | 261.8 | rationale mis-multiplied 12*0.4*16 as 384 (true 76.8); stored sum 554 matches neither |
| S3-0197 | 270.0 | rationale mis-multiplied 40*1.8*2 as 396 (true 144) |
| S3-0287 | 27.29 | terms sum to 655 but stored answer used 705/24 |
| S3-0292 | 24.58 | 590/24 = 24.583, not 23.25 |
| S5-0240 | 1290.0 | AF difference 0.1306-0.0769=0.0537; 24,000*0.0537=1290, not 1920 |
| S5-0286 | 500.0 | restores the _answer_corrected_v3 marker value (500) that a May 28 edit reverted to 50; models answered 500 in all 28 cells |
| S5-0292 | 285.5 | rationale's own AF 0.0951 gives 3,000*0.0951=285.5, not 659 |
| S5-0326 | 846.0 | rationale's own AF 0.1174 gives 7,200*0.1174=846, not 1872 |
| S5-0331 | 1583.0 | stored 16,800 implies AF=1.2>1 (physically impossible); 14,000*0.1131=1583 |
| S5-0334 | 528.8 | rationale's own AF 0.1173 gives 4,500*0.1173=528.8, not 2240.96 |

## Round-2 gold-answer fixes (37 — from the full verification sweep: 311 calculation golds independently recomputed, every flag adversarially verified by a second independent recomputation; 7 flags were refuted and kept unchanged)

| qid | new gold | unit | reason (abridged) |
|---|---|---|---|
| S1-0008 | 29.2 | m³/day | Independent recompute: 4 h × 2.5 m³/h = 10.0 and 4 h × 4.8 m³/h = 19.2, total 29.2 m³. Stored 30.8 is 5.48% off (tolerance 0.05), and the ra |
| S1-0104 | 18.08 | m³/day | The stored 15.3 contradicts its own rationale and is unreachable by any consistent route. Using the bench's own authoritative KB (knowledge_ |
| S2-0341 | 35 | ppb | With no indoor sources or removal (k=0), steady state gives C_in = P·AER·Cout/(AER+0) = P·Cout = 0.7×50 = 35 ppb. Stored 28 = 0.7×0.8×50 mul |
| S2-0342 | 32 | ppb | With no ozone removal indoors, C_in = P·AER·Cout/(AER+0) = P·Cout = 0.4×80 = 32 ppb. Stored 48 = 80×0.4×1.5 applies the spurious AER factor  |
| S2-0345 | 60 | μg/m³ | Steady-state mass balance gives C_in = P·AER·C_out/(AER+k); the question explicitly states 'no indoor sources or deposition exist', forcing  |
| S2-0370 | 20 | ppb | Question states 'no indoor sources or sinks', so k=0 and C_in = P·AER·C_out/(AER+k) = P·C_out = 0.4×50 = 20 ppb. Stored 30.0 = 0.4×50×1.5 us |
| S2-0371 | 12 | ppb | Same dimensionally invalid formula; correct steady-state no-sink result is C_in = P·C_out = 0.2×60 = 12 ppb. Stored 6.0 would require an ind |
| S2-0372 | 60 | μg/m³ | Stored gold uses C_in = P·C_out·AER (0.8×75×0.6=36), which is dimensionally invalid — the item's own gold reference (REF_WHO_AQG_2021) gives |
| S2-0373 | 36 | ppb | Question explicitly states 'no indoor sources or sinks', so k=0 and steady state is C_in = P·C_out = 0.9×40 = 36 ppb per the item's own WHO  |
| S3-0188 | 18.96 | μg/m³ | Hours sum to exactly 24; Σ(t·C) = 5×10+3×35+10×15+6×25 = 455; 455/24 = 18.958 ≈ 18.96. Stored 17.08 implies Σ = 410, unreachable under any r |
| S3-0200 | 15.67 | μg/m³ | Question explicitly averages over the stated 12 hours (4+2+6 = 12); Σ(t·C) = 40+100+48 = 188; 188/12 = 15.667 ≈ 15.67. Stored 17.33 implies  |
| S3-0276 | 22.25 | μg/m³ | Hours sum to exactly 24; Σ(t·C) = 108+176+130+120 = 534; 534/24 = 22.25 exactly (within the 0.01 tolerance). Stored 24.25 implies Σ = 582, u |
| S3-0277 | 25.83 | μg/m³ | TWA = (5×30 + 12×10 + 7×50)/24 = (150+120+350)/24 = 620/24 = 25.8333 μg/m³. The stored rationale itself writes '620/24 = 22.92', an arithmet |
| S3-0280 | 28.5 | μg/m³ | Independent recomputation: Σ(t·C) = 7×12 + 10×25 + 3×70 + 4×35 = 84+250+210+140 = 684 μg·h/m³; Σt = 24 h exactly, so TWA = 684/24 = 28.5 μg/ |
| S3-0282 | 31.67 | μg/m³ | Σ(t·C)=4·15+8·20+4·55+8·40=760 and Σt=24 h exactly, so TWA=760/24=31.6667. Stored 32.92 implies a numerator of 790.08, which no reading of t |
| S3-0283 | 21.67 | μg/m³ | 12·12+4·50+8·22=520 with Σt=24 h exactly, so TWA=520/24=21.6667. Stored 21.33 corresponds to 512/24, an arithmetic slip; the rationale state |
| S3-0284 | 20.54 | μg/m³ | 8·12+9·18+4·40+3·25=493 with Σt=24 h exactly, so TWA=493/24=20.5417. Stored 20.25 equals 486/24; no subset or variant of the four stated mic |
| S3-0286 | 23.5 | μg/m³ | TWA = (10×12 + 8×18 + 6×50)/24 = 564/24 = 23.5 exactly. Stored gold 21.5 is an arithmetic error in the rationale itself, which writes (120+1 |
| S3-0291 | 26.25 | μg/m³ | TWA = (8×10 + 10×25 + 6×50)/24 = 630/24 = 26.25 exactly. The rationale correctly derives 630/24 but states the quotient as 25.83 (which woul |
| S3-0294 | 24.54 | μg/m³ | TWA = (7×12 + 8×20 + 3×55 + 6×30)/24 = 589/24 = 24.5417. The rationale correctly sums to 589 but misdivides to 22.58 (which would be ~542/24 |
| S3-0295 | 17.83 | μg/m³ | Hours sum to 24; item's own rationale gives integrated exposure 428 μg·h/m³, and 428/24 = 17.8333, not 19.83. Pure final-division arithmetic |
| S3-0296 | 22.92 | μg/m³ | Hours sum to 24; rationale's integrated sum 550 is correct (100+90+120+240), but 550/24 = 22.9167, not 20.33. No alternate reading reproduce |
| S3-0301 | 22.58 | μg/m³ | Hours sum to 24; rationale's integrated sum 542 is correct (105+225+140+72), but 542/24 = 22.5833, not 24.58. Question is well-posed and una |
| S3-0302 | 21.75 | μg/m³ | Well-posed: hours sum to 24. Sum of t·C = 90+144+180+108 = 522 (matches the item's own rationale intermediates); 522/24 = 21.75. Stored 21.4 |
| S3-0304 | 19.0417 | μg/m³ | Well-posed: hours sum to 24. Sum of t·C = 60+176+165+56 = 457 (matches rationale intermediates); 457/24 = 19.0417. Stored 20.25 (=486/24) ma |
| S3-0305 | 22.125 | μg/m³ | Well-posed: hours sum to 24. Rationale itself writes '531/24 = 20.625', a bare arithmetic error: 531/24 = 22.125. Stored gold (=495/24) is n |
| S4-0129 | 0.03 | unitless | Independent recompute: CSF 1.5 per mg/kg/day * LADD 0.02 mg/kg/day = 0.03, dimensionless. Numeric gold is correct and unchanged; only the un |
| S4-0131 | 2.857e-06 | mg/kg/day | Independent recompute: (2e-5 mg/m3 * 20 m3/day * 0.5 * 365 * 70)/(70 kg * 25550 day) = 2.8571e-06 mg/kg/day. Stored answer 2.86e-06 paired w |
| S4-0136 | 2.571e-07 | unitless | Independent recompute: 1.8e-3 per µg/m3 * 0.001 µg/m3 * (10/70) * (365/365) * (24/24) = 2.5714e-07, dimensionless. Stored unit '×10⁻³' is th |
| S5-0193 | 117.5 | deaths/year | Independent recomputation: dC = 30-5 = 25 ug/m3; RR = exp(0.005*25) = 1.133148; AF = (RR-1)/RR = 0.117503 (identically 1-exp(-b*dC)); deaths |
| S5-0196 | 6593.6 | cases/year | Independent recomputation: dC = 40-20 = 20 ppb; RR = exp(0.02*20) = exp(0.4) = 1.491825; AF = (RR-1)/RR = 0.329680; cases = 2,000,000 x 0.01 |
| S5-0212 | 1262.8 | deaths/year | Stored 1350 is the linear shortcut pop*rate*beta*dC, contradicting the item's own rationale (RR=exp(0.135)=1.1445, AF=(RR-1)/RR=0.1263 -> 2e |
| S5-0215 | 146.3 | deaths/year | Question explicitly pins RR=exp(0.005) per ug/m3 and the attributable fraction approach: RR=exp(0.05)=1.0513, AF=0.04877, deaths=1e6*0.003*0 |
| S5-0223 | 638.4 | cases/year | Stored 675 equals the linear shortcut AF=beta*dC=0.1125, but the rationale states AF=(RR-1)/RR (=0.1064 with RR=exp(0.1125)=1.1191), giving  |
| S5-0237 | 3173 | deaths/year | DeltaC = 25 is unambiguous and the rationale explicitly commits to the log-linear PAF method: RR = exp(0.005 x 25) = 1.13315, AF = (RR-1)/RR |
| S5-0244 | 256 | DALYs/year | Baseline DALYs = 1,500,000 x 400/100,000 = 6,000. DeltaAF = (1 - exp(-0.003 x 25)) - (1 - exp(-0.003 x 10)) = 0.072257 - 0.029554 = 0.042702 |
| S5-0298 | 1.419 | dimensionless | Question explicitly gives RR=exp(beta*dC) with beta=0.014, dC=25: exp(0.35)=1.4191 (internally consistent beta=ln(1.15)/10 gives 1.4182, wit |

## Round-2 retirements (4)

- **S1-0172**: provenance-compromised: the question wording was edited on May 28 after 19 of 28 cells had already been served the old wording; scored results are not reproducible from the current bank text
- **S1-0246**: 3.0 L/day is not defensible under any reading: the item's own gold_reference quote states 'the average adult intake rate is about 1.15 L/day'; the bench KB and the item's own gold tool call (drinking_water, age 21, sex M, long_term) both return 1,053
- **S2-0226**: Under-specified: the model C=(P·AER·Cout+S/V)/(AER+k) requires room volume V, but the question gives S=15 μg/h with no V, so the answer is indeterminate (V=50 m³ gives ~23.6 μg/m³). Even the charitable volumetric reading (V=1 m³) gives (0.7·1·40+15)/
- **S3-0219**: The two segments with stated inhalation rates give walking 30×1.2×2 = 72 μg and car 50×0.8×3 = 120 μg. The 5 h home segment is given a concentration (8 μg/m³) but no inhalation rate, and the gold of 192 μg is obtainable only by setting IR_home = 0 — 

## Unit-field repairs (2 — answer values unchanged)

- **S1-0014**: unit → `fraction` (unit field was '(91%)' for a dimensionless fraction (answer 0.91))
- **S3-0156**: unit → `µg/m³` (unit field contained the entire worked solution string)

## Retired: defective items (6)

- **S1-0248**: ambiguous: 'a 3-year-old' straddles the EFH 2-<3 and 3-<6 surface-area bins; gold 0.66 matches neither published bin value; tolerance 1%
- **S1-0249**: gold unverifiable (700 g/day not confirmed against EFH; archived answers cluster 1000-1290 g/day) and tolerance corrupted (10, i.e. any answer within +/-7000 scored 1.0)
- **S3-0234**: under-specified: activity-level inhalation rates not given in question; gold assumes 4.8/0.54 m3/h while the item's own cited EFH table gives 4.2/0.498 (TWA 1.42 vs gold 1.62)
- **S4-0134**: under-specified: ED/AT not stated; rationale's inhalation HQ (0.33) irreproducible from its own formula; archived model answers scatter 0.40-0.92 with no defensible modal value
- **S4-0138**: under-specified: averaging time not stated; stored gold 5.43e-08 irreproducible; the rationale's own formula yields 1.03e-3
- **S5-0253**: ill-posed: 'excess risk' as (RR-1)*exposure is not a defined IER quantity; rationale arithmetic inconsistent ((1.14-1)*20 stated as 0.28); answer carries a concentration unit for a risk quantity

## Retired: TF-balance (68 True-labeled items; author decision)

Key rebalanced to 92 True / 92 False (TF n 252 → 184). Deterministic selection: 
over-represented tiers only (simple, then medium); 26 of 68 were items 
solved by all 20 paper cells (zero discrimination — cheapest to remove); the rest 
seeded-random (seed 42). qids:

```
S1-0125, S1-0130, S1-0143, S1-0147, S1-0150, S1-0158, S1-0172, S1-0178, S1-0182, S1-0191, S1-0275, S2-0205, S2-0220, S2-0225, S2-0232, S2-0247, S2-0251, S2-0253, S2-0263, S2-0267, S2-0272, S2-0274, S2-0277, S2-0355, S3-0045, S3-0167, S3-0170, S3-0173, S3-0177, S3-0180, S3-0192, S3-0199, S3-0203, S3-0206, S3-0212, S3-0215, S3-0221, S3-0225, S3-0228, S3-0236, S3-0312, S3-0315, S3-0318, S4-0133, S4-0146, S4-0152, S4-0159, S4-0163, S4-0172, S4-0191, S4-0203, S4-0208, S4-0212, S4-0215, S4-0286, S5-0170, S5-0173, S5-0186, S5-0198, S5-0201, S5-0202, S5-0219, S5-0225, S5-0233, S5-0242, S5-0246, S5-0252, S5-0255
```

## Content flags (no score impact; reword in a future revision)

- **S1-0256**: question embeds a physically impossible SA factor (0.25 m2/kg; realistic ~0.034), so gold 8.0 m2 follows from the question but is nonsense; internally consistent, gold left unchanged for score stability; reword in a future revision

Note: S1-0254 needed no bank change — its gold is correct; the false zeros were
a scorer unit-normalization defect fixed in `rescore_v2_20260811.py`.
