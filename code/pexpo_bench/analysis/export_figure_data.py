"""Export the underlying data for every figure into one multi-sheet Excel.
Output: article/final/PExpo-Bench_figure_data_v4.xlsx
"""
import os
import json, pathlib
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
df = pd.read_parquet(ROOT / 'runs/v4_scored/all_scored_v4_main.parquet')
OUT = ROOT / 'article/final/PExpo-Bench_figure_data_v4.xlsx'

PAPER = {'A0_naive': 'A0', 'A1_context_eng': 'A1', 'A2p_rag_constrained': 'A2',
         'A3_agent': 'A3', 'A4p_hybrid_constrained': 'A4'}
MODELS = ['gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'deepseek-v4']
ORDER = ['A0', 'A1', 'A2', 'A3', 'A4']
COST = {'gpt-5.4': (5e-3, 15e-3), 'gpt-5.4-mini': (2.5e-4, 2.0e-3),
        'gpt-5.4-nano': (1.5e-4, 6e-4), 'deepseek-v4': (2.7e-4, 1.1e-3)}

d = df[df.arch.isin(PAPER)].copy(); d['A'] = d.arch.map(PAPER)

def boot_ci(v, n=500, seed=42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

# ---- Fig 2a overall accuracy + 95% CI ----
rows = []
for m in MODELS:
    for a in ORDER:
        v = d[(d.model == m) & (d.A == a)]['score'].values
        lo, hi = boot_ci(v)
        rows.append(dict(model=m, architecture=a, n=len(v),
                         accuracy_pct=round(v.mean()*100, 2),
                         ci95_low_pct=round(lo*100, 2), ci95_high_pct=round(hi*100, 2)))
fig2a = pd.DataFrame(rows)

# ---- Fig 2b accuracy vs cost ----
def cost_per_100q(m, a):
    cin, cout = COST[m]
    g = d[(d.model == m) & (d.A == a)]
    return ((g.in_tokens*cin + g.out_tokens*cout)/1000).mean()*100
fig2b = pd.DataFrame([dict(model=m, architecture=a,
                           accuracy_pct=round(d[(d.model==m)&(d.A==a)].score.mean()*100, 2),
                           cost_usd_per_100q=round(cost_per_100q(m, a), 3))
                      for m in MODELS for a in ORDER])

# ---- Fig 3 sub-domain accuracy ----
fig3 = (d.groupby(['model', 'A', 'subdomain'])['score'].mean()*100).round(2).reset_index()
fig3.columns = ['model', 'architecture', 'subdomain', 'accuracy_pct']

# ---- Fig 4 reliability: strict HR, accuracy, instruction-following ----
hr = pd.DataFrame([json.loads(l) for l in (ROOT/'runs/v4_rerun/_hr/per_row_hr.jsonl').read_text().splitlines() if l.strip()])
_sub = set((ROOT/'runs/v4_rerun/_hr/grounding_subsample_qids.txt').read_text().split())
hr = hr[hr.qid.isin(_sub)]
hr = hr[hr.arch.isin(PAPER)].copy(); hr['A'] = hr.arch.map(PAPER)
rows = []
for m in MODELS:
    for a in ORDER:
        g = d[(d.model == m) & (d.A == a)]
        h = hr[(hr.model == m) & (hr.A == a)]
        sHR = h.n_contradicted.sum()/h.n_claims.sum()*100 if h.n_claims.sum() else np.nan
        rows.append(dict(model=m, architecture=a,
                         strict_HR_pct=round(sHR, 3),
                         accuracy_pct=round(g.score.mean()*100, 2),
                         instruction_following_pct=round((1-g.parse_error.mean())*100, 2)))
fig4 = pd.DataFrame(rows)

# ---- Fig 5 harness payoff A4 - A3 ----
pmap = {'gpt-5.4': 7.5e-5, 'gpt-5.4-mini': 0.53, 'gpt-5.4-nano': 4.9e-37, 'deepseek-v4': 6.6e-4}
def star(p): return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'
fig5 = pd.DataFrame([dict(model=m,
                          A3_acc_pct=round(d[(d.model==m)&(d.A=='A3')].score.mean()*100, 2),
                          A4_acc_pct=round(d[(d.model==m)&(d.A=='A4')].score.mean()*100, 2),
                          delta_A4_minus_A3_pp=round((d[(d.model==m)&(d.A=='A4')].score.mean()-d[(d.model==m)&(d.A=='A3')].score.mean())*100, 2),
                          wilcoxon_p=pmap[m], significance=star(pmap[m])) for m in MODELS])

# ---- Fig S1 efficiency x robustness ----
rows = []
for m in MODELS:
    for a in ORDER:
        g = d[(d.model == m) & (d.A == a)]
        tot = g.score.sum()
        tpc = (g.in_tokens.sum()+g.out_tokens.sum())/tot if tot else np.nan
        rows.append(dict(model=m, architecture=a,
                         tokens_per_correct=round(tpc, 1),
                         parse_error_pct=round(g.parse_error.mean()*100, 2)))
figS1 = pd.DataFrame(rows)

# ---- Fig S2 taxonomy: per-item number of architectures correct (k) ----
rows = []
for m in MODELS:
    sub = d[d.model == m]
    piv = sub.pivot_table(index='qid', columns='A', values='score', aggfunc='first')
    meta = sub.drop_duplicates('qid').set_index('qid')[['subdomain', 'question_type', 'difficulty']]
    k = (piv[ORDER] >= 0.5).sum(axis=1)
    for qid in k.index:
        rows.append(dict(model=m, qid=qid, subdomain=meta.loc[qid, 'subdomain'],
                         question_type=meta.loc[qid, 'question_type'],
                         difficulty=meta.loc[qid, 'difficulty'], n_archs_correct=int(k[qid])))
figS2 = pd.DataFrame(rows)

# ---- (evidence-use ablation dropped: its plain arms exist only in the original
# campaign; the corrected rerun's factorial decomposition has its own generator
# make_fig_s2.py reading the phase-B arms) ----

# ---- Fig S4 mechanism (from reproducible summary) ----
summ = json.loads((ROOT/'runs/v4_rerun/_mechanism/summary.json').read_text())
figS4 = pd.DataFrame([dict(model=m, **summ[m]) for m in MODELS])

readme = pd.DataFrame([
    ('Fig2a_overall_accuracy', 'Figure 2a: overall accuracy + 95% bootstrap CI per model x architecture'),
    ('Fig2b_accuracy_cost', 'Figure 2b: accuracy vs cost (USD per 100 questions) per cell'),
    ('Fig3_subdomain_accuracy', 'Figure 3: accuracy per model x architecture x sub-domain'),
    ('Fig4a_reliability', 'Figure 4a: strict hallucination rate, accuracy, instruction-following per cell'),
    ('Fig4b_harness_payoff', 'Figure 4b: A4 minus A3 harness payoff per model with Wilcoxon p'),
    ('FigS1_efficiency', 'Figure S1: tokens per correct answer and parse-error rate per cell'),
    ('FigS2_mechanism', 'Figure S2 / Table S4: control-flow metrics A3 vs A4 per model'),
    ('FigS3_taxonomy', 'Figure S3: per-item count of architectures answering correctly (k, 0-5)'),
    ('note', 'Figure 1 is a schematic of the architectures and has no underlying data.'),
], columns=['sheet', 'description'])

with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    readme.to_excel(w, sheet_name='README', index=False)
    fig2a.to_excel(w, sheet_name='Fig2a_overall_accuracy', index=False)
    fig2b.to_excel(w, sheet_name='Fig2b_accuracy_cost', index=False)
    fig3.to_excel(w, sheet_name='Fig3_subdomain_accuracy', index=False)
    fig4.to_excel(w, sheet_name='Fig4a_reliability', index=False)
    fig5.to_excel(w, sheet_name='Fig4b_harness_payoff', index=False)
    figS1.to_excel(w, sheet_name='FigS1_efficiency', index=False)
    figS4.to_excel(w, sheet_name='FigS2_mechanism', index=False)
    figS2.to_excel(w, sheet_name='FigS3_taxonomy', index=False)
print('wrote', OUT)
for nm, fr in [('Fig2a', fig2a), ('Fig2b', fig2b), ('Fig3', fig3), ('Fig4a', fig4),
               ('Fig4b', fig5), ('FigS1', figS1), ('FigS2', figS4), ('FigS3', figS2)]:
    print(f'  {nm}: {fr.shape[0]} rows')
