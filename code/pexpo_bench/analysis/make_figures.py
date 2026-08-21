""" figures (5-arch edition).

Main figures (article/final/svg-fig-v3/):
  Fig 1 Architecture mechanism component matrix
  Fig 2 Accuracy bars + Pareto frontier (3 base models)
  Fig 3 Sub-domain accuracy lollipop (3 base models)
  Fig 4 Reliability (HR_RefChecker lollipop + instruction-following + scatter)

SI figures:
  Fig S1 Difficulty taxonomy (3 base models)
  Fig S2 Efficiency × robustness scatter

Architectures: A0 naive, A1 static-context, A2 RAG (rules baked in),
               A3 tool agent, A4 hybrid (rules baked in).
"""
import os
import json, pathlib, math, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle
from scipy.stats import wilcoxon

from paper_palette import (
    ACC_CMAP,
    ALERT,
    ARCH_EDGE as EDGE,
    ARCH_FILL as FILL,
    COMPONENT_STYLE,
    DIV_CMAP,
    FRAME,
    MODEL_COLOR,
    ROSE_CMAP,
)

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
PARQ = ROOT / 'runs/v4_scored/all_scored_v4_main.parquet'
HR_FILE = ROOT / 'runs/v4_rerun/_hr/per_row_hr.jsonl'
RUNS_DIR = ROOT / 'runs/v4_rerun'
TOOL_RUNS_DIR = ROOT / 'runs/_none_v4'
OUT = ROOT / 'article/final/svg-fig-v4'
OUT.mkdir(parents=True, exist_ok=True)

# 5-arch mapping: A2+ folder becomes A2; A4+ folder becomes A4; old A2/A4 dropped.
ARCHS = ['A0','A1','A2','A3','A4']
DIR2ARCH = {'A0_naive':'A0', 'A1_context_eng':'A1',
            'A2p_rag_constrained':'A2', 'A3_agent':'A3',
            'A4p_hybrid_constrained':'A4'}
ARCH_DESC = {'A0':'A0 naive', 'A1':'A1 static context', 'A2':'A2 RAG',
             'A3':'A3 tool agent', 'A4':'A4 harness'}
MARKER = {'A0':'o', 'A1':'s', 'A2':'^', 'A3':'D', 'A4':'v'}

MODELS = ['GPT-5.4', 'GPT-5.4-mini', 'GPT-5.4-nano', 'DeepSeek-V4']
MODEL_KEY = {'GPT-5.4':'gpt-5.4', 'GPT-5.4-mini':'gpt-5.4-mini',
             'GPT-5.4-nano':'gpt-5.4-nano', 'DeepSeek-V4':'deepseek-v4'}
KEY2MODEL = {v:k for k,v in MODEL_KEY.items()}
plt.rcParams.update({
    'font.family':'sans-serif',
    'font.size':10,
    'axes.linewidth':0.8,
    'axes.edgecolor':'#333333',
    'axes.labelcolor':'#222222',
    'text.color':'#222222',
    'xtick.color':'#333333', 'ytick.color':'#333333',
    'xtick.direction':'out', 'ytick.direction':'out',
    'xtick.major.size':3.2, 'ytick.major.size':3.2,
    'xtick.major.width':0.8, 'ytick.major.width':0.8,
    'axes.titlepad':7,
})

# =================================================================
# Data loading
# =================================================================
def _load():
    df = pd.read_parquet(PARQ)
    df = df[df['arch'].isin(DIR2ARCH)].copy()
    df['arch'] = df['arch'].map(DIR2ARCH)
    df['model'] = df['model'].map(KEY2MODEL)
    return df

LONG = _load()

COST_PER_1K = {'GPT-5.4':{'in':5e-3,'out':15e-3},
               'GPT-5.4-mini':{'in':2.5e-4,'out':2.0e-3},
               'GPT-5.4-nano':{'in':1.5e-4,'out':6e-4},
               'DeepSeek-V4':{'in':2.7e-4,'out':1.1e-3}}
def _cost(m, in_t, out_t):
    c = COST_PER_1K[m]
    return (in_t/1000)*c['in'] + (out_t/1000)*c['out']
LONG['cost_q'] = LONG.apply(lambda r: _cost(r['model'], r['in_tokens'], r['out_tokens']), axis=1)

ACC = {m: {a: LONG[(LONG.model==m)&(LONG.arch==a)]['score'].mean() for a in ARCHS} for m in MODELS}
COST = {m: {a: LONG[(LONG.model==m)&(LONG.arch==a)]['cost_q'].mean()*100 for a in ARCHS} for m in MODELS}

def _wilcoxon_p(model, a, b):
    sub = LONG[LONG.model==model]
    piv = sub.pivot_table(index='qid', columns='arch', values='score', aggfunc='first')
    try:
        _, p = wilcoxon(piv[a], piv[b], zero_method='wilcox')
        return p
    except Exception:
        return float('nan')

def _pfmt(p):
    if np.isnan(p): return 'ns'
    if p < 1e-4: return 'p<10⁻⁴'
    if p < 1e-3: return f'p<10⁻³'
    if p < 0.05: return f'p={p:.3f}'
    return f'p={p:.2f}'

def _bootstrap_ci(vals, n=500, seed=42):
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n)])
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

def _spines_box(ax):
    """Full four-sided frame, thin and dark grey."""
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color(FRAME)

# =================================================================
# Fig 1 — Architecture design space (left: Venn) + pipelines (right)
# =================================================================
def fig_1_arch_mechanism():
    from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

    fig = plt.figure(figsize=(17.5, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.02,
                          left=0.02, right=0.99, top=0.94, bottom=0.06)

    # ---------- LEFT: Venn ----------
    axL = fig.add_subplot(gs[0])
    axL.set_xlim(-3.7, 3.7); axL.set_ylim(-4.5, 3.4)
    axL.set_aspect('equal'); axL.axis('off')

    r = 1.95
    cx_S, cy_S =  0.00,  0.65
    cx_R, cy_R = -1.00, -0.85
    cx_T, cy_T =  1.00, -0.85
    set_col_S = COMPONENT_STYLE['ctx'][1]
    set_col_R = COMPONENT_STYLE['ret'][1]
    set_col_T = COMPONENT_STYLE['tool'][1]

    for (cx, cy), col in [((cx_S, cy_S), set_col_S),
                          ((cx_R, cy_R), set_col_R),
                          ((cx_T, cy_T), set_col_T)]:
        axL.add_patch(Circle((cx, cy), r, fc=col, ec='none', alpha=0.08, zorder=1))
        axL.add_patch(Circle((cx, cy), r, fc='none', ec=col, lw=1.2,
                             linestyle=(0,(4,2)), zorder=2))

    axL.text(0.00, 2.95, 'Static context  ·  EFH handbook in prompt',
             ha='center', fontsize=11.5, fontweight='bold', color=set_col_S)
    axL.text(-3.40, -2.55, 'Retrieval  ·  top-5 + evidence rules',
             ha='center', fontsize=11.5, fontweight='bold', color=set_col_R)
    axL.text(3.40, -2.55, 'Tool calls  ·  17-function registry',
             ha='center', fontsize=11.5, fontweight='bold', color=set_col_T)

    placements = {
        'A0': (0.00, -3.85, 'naive'),
        'A1': (0.00,  1.55, 'static context'),
        'A2': (-1.55, -0.30, 'RAG'),
        'A3': (1.55, -1.05, 'tool agent'),
        'A4': (0.00, -0.85, 'hybrid'),
    }
    for arch, (x, y, lbl) in placements.items():
        is_hybrid = (arch == 'A4')
        size = 1100 if is_hybrid else 850
        axL.scatter(x, y, s=size, color=FILL[arch], edgecolor=EDGE[arch],
                    lw=2.3 if is_hybrid else 1.8, zorder=5)
        axL.text(x, y + 0.06, arch, ha='center', va='center',
                 fontsize=16 if is_hybrid else 14, fontweight='bold',
                 color=EDGE[arch], zorder=6)
        axL.text(x, y - (0.50 if is_hybrid else 0.42), lbl, ha='center', va='center',
                 fontsize=10, color='#2a2a2a', zorder=6)
    axL.set_title('a', loc='left', fontsize=12, fontweight='bold', pad=6)

    # ---------- RIGHT: pipeline schematics ----------
    axR = fig.add_subplot(gs[1])
    axR.set_xlim(0, 14.5); axR.set_ylim(0.4, 8.0)
    axR.set_aspect('auto'); axR.axis('off')

    STAGE_COL = COMPONENT_STYLE

    def stage(x, y, w, h, label, kind, bold=False, fontsize=8.8):
        fc, ec = STAGE_COL[kind]
        axR.add_patch(FancyBboxPatch((x, y-h/2), w, h,
                                     boxstyle='round,pad=0.02,rounding_size=0.10',
                                     fc=fc, ec=ec, lw=1.3, zorder=3))
        axR.text(x+w/2, y, label, ha='center', va='center',
                 fontsize=fontsize, fontweight='bold' if bold else 'normal',
                 color=ec, zorder=4)
        return x+w

    def arrow(x1, y1, x2, y2, bidir=False):
        if bidir:
            axR.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                          arrowstyle='<|-|>', mutation_scale=12,
                                          color=EDGE['A3'], lw=1.6, zorder=2))
        else:
            axR.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                          arrowstyle='-|>', mutation_scale=10,
                                          color='#666', lw=1.1, zorder=2))

    rows = [
        ('A0', ['io:Q', 'llm:LLM', 'io:Ans']),
        ('A1', ['ctx:EFH', 'io:Q', 'llm:LLM', 'io:Ans']),
        ('A2', ['ctx:EFH', 'ret:RAG top-5', 'io:Q', 'llm:LLM', 'io:Ans']),
        ('A3', ['io:Q', 'llm:LLM', 'tool:17 funcs', 'io:Ans']),
        ('A4', ['ctx:EFH', 'ret:RAG top-5', 'io:Q', 'llm:LLM', 'tool:18 funcs', 'io:Ans']),
    ]
    ys = [7.30, 6.00, 4.65, 3.20, 1.55]

    H_DEF = 0.50
    W = {'io':0.72, 'ctx':1.05, 'ret':1.20, 'tool':1.55, 'llm':0.92}

    for (arch, stages), y in zip(rows, ys):
        # arch badge — use scatter so circle is perfectly round regardless of axis aspect
        axR.scatter(0.55, y, s=820, color=FILL[arch], edgecolor=EDGE[arch],
                    lw=1.9, zorder=5)
        axR.text(0.55, y+0.01, arch, ha='center', va='center',
                 fontweight='bold', fontsize=11.5, color=EDGE[arch], zorder=6)
        axR.text(1.10, y, ARCH_DESC[arch].split(' ', 1)[1],
                 ha='left', va='center', fontsize=10, color='#2a2a2a', fontweight='bold')

        x = 3.20
        prev_x = None
        for st in stages:
            kind, lbl = st.split(':', 1)
            w = W[kind]
            h = H_DEF
            if prev_x is not None:
                # bidir between LLM and tool
                bidir = (kind == 'tool' or (prev_kind == 'tool' and kind == 'io'
                                            and len(stages) > stages.index(st) - 1))
                bidir = (kind == 'tool')
                arrow(prev_x + 0.05, y, x - 0.05, y, bidir=bidir)
            stage(x, y, w, h, lbl, kind,
                  bold=(kind == 'llm'),
                  fontsize=9.0 if kind in ('llm','io') else 8.6)
            prev_x = x + w
            prev_kind = kind
            x = prev_x + 0.25

        # step-budget note for tool archs
        if arch == 'A3':
            axR.text(x + 0.4, y, '≤ 8 steps', ha='left', va='center',
                     fontsize=8.5, color='#6c6c6c', style='italic')
        elif arch == 'A4':
            axR.text(x + 0.4, y, '≤ 10 steps', ha='left', va='center',
                     fontsize=8.5, color='#6c6c6c', style='italic')

    # header strip (no separator line — keeps panel cleaner)
    pass

    axR.set_title('b', loc='left', fontsize=12, fontweight='bold', pad=6)

    p1 = OUT/'Figure_1_arch_mechanism.png'; p2 = OUT/'Figure_1_arch_mechanism.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig 1 → {p1.name}')

# =================================================================
# Fig 2 — accuracy bars (top) + Pareto (bottom)
# =================================================================
def fig_2_acc_pareto():
    # Merged results landscape:
    #   (a) accuracy heatmap        base model (rows) x architecture (cols)
    #   (b) sub-domain diverging heatmap   4 models x 2 tool arches (rows) x sub-domain (cols)
    #   (c) global cost-accuracy Pareto over all 20 cells (open-weight vs proprietary)
    fig = plt.figure(figsize=(14.4, 9.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.28,
                             left=0.115, right=0.955, top=0.92, bottom=0.13)
    left = outer[0].subgridspec(2, 1, height_ratios=[4, 8], hspace=0.92)
    axa = fig.add_subplot(left[0]); axb = fig.add_subplot(left[1])
    axc = fig.add_subplot(outer[1])

    # --- panel a: accuracy heatmap (models x architectures) ---
    row_models = ['DeepSeek-V4', 'GPT-5.4', 'GPT-5.4-mini', 'GPT-5.4-nano']
    Ma = np.array([[ACC[m][a]*100 for a in ARCHS] for m in row_models])
    im_a = axa.imshow(Ma, cmap=ACC_CMAP, vmin=45, vmax=90, aspect='auto')
    for i in range(Ma.shape[0]):
        for j in range(Ma.shape[1]):
            v = Ma[i, j]
            axa.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=10.5,
                     fontweight='bold', color='white' if v > 72 else '#2a2a2a')
    axa.set_xticks(range(len(ARCHS))); axa.set_xticklabels([ARCH_DESC[a] for a in ARCHS], fontsize=9)
    axa.xaxis.set_ticks_position('top')
    axa.set_yticks(range(len(row_models))); axa.set_yticklabels(row_models, fontsize=9.5, fontweight='bold')
    axa.tick_params(length=0)
    axa.set_xticks(np.arange(-.5, len(ARCHS), 1), minor=True)
    axa.set_yticks(np.arange(-.5, len(row_models), 1), minor=True)
    axa.grid(which='minor', color='white', linewidth=2.4); axa.tick_params(which='minor', length=0)
    axa.axvline(2.5, color='#bcbcbc', lw=1.0, zorder=4)   # no-tool | tool architecture split
    tr = axa.get_xaxis_transform()
    for x0, x1, lab in [(-0.44, 2.44, 'No-tool architectures'),
                        (2.56, 4.44, 'Tool-enabled architectures')]:
        axa.plot([x0, x1], [1.135, 1.135], color='#999999', lw=1.0, transform=tr, clip_on=False)
        axa.plot([x0, x0], [1.135, 1.10], color='#999999', lw=1.0, transform=tr, clip_on=False)
        axa.plot([x1, x1], [1.135, 1.10], color='#999999', lw=1.0, transform=tr, clip_on=False)
        axa.text((x0+x1)/2, 1.165, lab, transform=tr, ha='center', va='bottom',
                 fontsize=8, color='#555555')
    _spines_box(axa)
    caxa = axa.inset_axes([0.20, -0.28, 0.60, 0.05])
    cba = fig.colorbar(im_a, cax=caxa, orientation='horizontal')
    cba.set_label('Mean accuracy (%)', fontsize=8); cba.ax.tick_params(length=2.5, labelsize=7.5)
    cba.outline.set_linewidth(0.8); cba.outline.set_edgecolor('#333333')
    axa.set_title('a', loc='left', fontweight='bold', fontsize=12, pad=44)

    # --- panel b: sub-domain diverging heatmap (model x tool arch, vs no-tool baseline) ---
    SD_MAP = {'S1_exposure_factors':'S1\nExposure\nfactors',
              'S2_microenv_conc':'S2\nMicroenvironment\nconcentration',
              'S3_trajectory_activity':'S3\nTrajectory/\nactivity', 'S4_dosimetry':'S4\nDosimetry',
              'S5_health':'S5\nHealth\neffects'}
    SD_KEYS = list(SD_MAP); BASE = ['A0', 'A1', 'A2']; TOOL = ['A3', 'A4']
    rows, rowlab = [], []
    for m in MODELS:
        piv = LONG[LONG.model==m].groupby(['subdomain','arch'])['score'].mean().unstack()\
                .reindex(SD_KEYS).reindex(ARCHS, axis=1)
        base = piv[BASE].mean(axis=1)
        for a in TOOL:
            rows.append([(piv.loc[sd, a]-base.loc[sd])*100 for sd in SD_KEYS]); rowlab.append(a)
    Mb = np.array(rows); VMAX = 25
    im_b = axb.imshow(Mb, cmap=DIV_CMAP, vmin=-VMAX, vmax=VMAX, aspect='auto')
    for i in range(Mb.shape[0]):
        for j in range(Mb.shape[1]):
            v = Mb[i, j]
            axb.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=8.5,
                     fontweight='bold', color='white' if abs(v) > 15 else '#2a2a2a')
    axb.set_xticks(range(len(SD_KEYS))); axb.set_xticklabels([SD_MAP[k] for k in SD_KEYS], fontsize=7.6)
    axb.xaxis.set_ticks_position('top')
    axb.set_yticks(range(len(rowlab))); axb.set_yticklabels(rowlab, fontsize=8.5)
    axb.tick_params(length=0)
    axb.set_xticks(np.arange(-.5, len(SD_KEYS), 1), minor=True)
    axb.set_yticks(np.arange(-.5, len(rowlab), 1), minor=True)
    axb.grid(which='minor', color='white', linewidth=2.2); axb.tick_params(which='minor', length=0)
    for g in range(1, 4):
        axb.axhline(2*g-0.5, color='#333333', lw=1.3)
    for gi, m in enumerate(MODELS):     # coloured bracket + name grouping each model's A3/A4 rows
        axb.plot([-0.62, -0.62], [2*gi-0.30, 2*gi+1.30], color=MODEL_COLOR[m], lw=1.8,
                 clip_on=False, zorder=5, solid_capstyle='round')
        axb.text(-0.80, 2*gi+0.5, m, fontsize=8.6, fontweight='bold', color=MODEL_COLOR[m],
                 ha='right', va='center')
    _spines_box(axb)
    cbb = fig.colorbar(im_b, ax=axb, fraction=0.022, pad=0.02)
    cbb.set_label('Change vs\nno-tool baseline (pp)', fontsize=8)
    cbb.ax.tick_params(length=2.5, labelsize=7.5)
    cbb.outline.set_linewidth(0.8); cbb.outline.set_edgecolor('#333333')
    axb.set_title('b', loc='left', fontweight='bold', fontsize=12, pad=22)

    # --- panel c: global cost-accuracy Pareto over all 20 cells ---
    # DeepSeek-V4 is open-weight; the GPT-5.4 family is proprietary. Both share one
    # global frontier, so the open-weight model visibly anchors the low-cost end while
    # the proprietary tool cells hold the high-accuracy end.
    BASE_MARKER = {'GPT-5.4':'o', 'GPT-5.4-mini':'P', 'GPT-5.4-nano':'s', 'DeepSeek-V4':'D'}
    OPEN = {'DeepSeek-V4'}
    cellz = [(COST[m][a], ACC[m][a]*100, m, a) for m in MODELS for a in ARCHS]
    def _dom(c, acc):
        return any((c2 <= c and acc2 >= acc and (c2 < c or acc2 > acc))
                   for c2, acc2, _, _ in cellz)
    fx, fy = zip(*sorted([(c, acc) for c, acc, m, a in cellz if not _dom(c, acc)]))
    axc.plot(fx, fy, drawstyle='steps-post', color='#1f5148', lw=1.8, zorder=2)
    # frontier cells carry a bold dark ring; dominated cells are faded. Shape = base model
    # (DeepSeek-V4 = diamond, open-weight), fill colour = architecture.
    for c, acc, m, a in cellz:
        on = not _dom(c, acc)
        axc.scatter(c, acc, s=125 if on else 55, c=FILL[a], marker=BASE_MARKER[m],
                    edgecolor='#2a2a2a' if on else 'none', linewidth=1.7 if on else 0,
                    alpha=1.0 if on else 0.5, zorder=5 if on else 3)
    axc.annotate('open-weight model\nanchors the cheap frontier',
                 (COST['DeepSeek-V4']['A0'], 85.1), (0.013, 79.5), fontsize=8.2,
                 color='#1f5148', ha='left', va='center',
                 arrowprops=dict(arrowstyle='-', color='#1f5148', lw=0.8))
    axc.annotate('GPT-5.4 naive: dominated\nby GPT-5.4-mini + tools',
                 (COST['GPT-5.4']['A0'], 74.1), (0.55, 67.5), fontsize=8.2,
                 color='#9c574b', ha='left', va='center',
                 arrowprops=dict(arrowstyle='-', color='#c0392b', lw=0.8))
    axc.set_xscale('log'); axc.set_ylim(44, 91)
    axc.set_xlabel('Cost (USD / 100 questions, log scale)', fontsize=10)
    axc.set_ylabel('Overall accuracy (%)', fontsize=10.5)
    axc.grid(alpha=0.22, ls=':'); _spines_box(axc)
    axc.set_title('c', loc='left', fontweight='bold', fontsize=12, pad=8)

    # unified bottom legend: architecture (colour) | base model (shape) | frontier status
    arch_h = [plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=FILL[a],
                         markeredgecolor=EDGE[a], markersize=9, markeredgewidth=1.1) for a in ARCHS]
    model_h = [plt.Line2D([0],[0], marker=BASE_MARKER[m], color='w', markerfacecolor='#cfcfcf',
                          markeredgecolor='#555', markersize=9, markeredgewidth=1.0) for m in MODELS]
    stat_h = [plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='#cfcfcf',
                         markeredgecolor='#2a2a2a', markersize=9, markeredgewidth=1.7),
              plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#cfcfcf',
                         markeredgecolor='none', markersize=8, alpha=0.5)]
    mlabels = [m + ('  (open-weight)' if m in OPEN else '') for m in MODELS]
    leg1 = fig.legend(arch_h, [ARCH_DESC[a] for a in ARCHS], loc='lower center',
                      bbox_to_anchor=(0.22, 0.003), ncol=5, frameon=False, fontsize=8.1,
                      title='architecture (fill colour)', title_fontsize=8.5)
    fig.add_artist(leg1)
    leg2 = fig.legend(model_h, mlabels, loc='lower center', bbox_to_anchor=(0.60, 0.003),
                      ncol=4, frameon=False, fontsize=8.1,
                      title='base model (shape)', title_fontsize=8.5)
    fig.add_artist(leg2)
    fig.legend(stat_h, ['Pareto-efficient (frontier)', 'Other (dominated)'], loc='lower center',
               bbox_to_anchor=(0.885, 0.003), ncol=1, frameon=False, fontsize=8.1,
               title='frontier status', title_fontsize=8.5)

    p1 = OUT/'Figure_2_results.png'; p2 = OUT/'Figure_2_results.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig 2 (merged) → {p1.name}')

# =================================================================
# Fig 5 — architecture landscape (conceptual 2D task space)
# =================================================================
def fig_5_landscape():
    """Deployment decision: the harness payoff (A4 minus A3) per base model.
    A4 never significantly beats A3; it is neutral on the strong models and
    harmful on the weaker ones, so the tool agent A3 is the safe default."""
    order = ['gpt-5.4-nano', 'deepseek-v4', 'gpt-5.4-mini', 'gpt-5.4']   # worst to best harness payoff
    labels = [KEY2MODEL[m] for m in order]
    delta = [(ACC[KEY2MODEL[m]]['A4'] - ACC[KEY2MODEL[m]]['A3']) * 100 for m in order]
    pval = {'gpt-5.4': 0.613, 'gpt-5.4-mini': 0.525, 'gpt-5.4-nano': 4.85e-37, 'deepseek-v4': 6.6e-4}
    def star(p): return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'
    fig, ax = plt.subplots(figsize=(10, 5.2))
    xmin, xmax = -27, 6
    ax.axvline(0, color='#444', lw=1.2, zorder=2)
    for i, (m, d) in enumerate(zip(order, delta)):
        sig = pval[m] < 0.05
        c = '#c08a8a' if sig else '#c9c9c9'          # red only if a significant loss; grey if ns
        ec = '#9e4b4b' if sig else '#777'
        ax.barh(i, d, color=c, edgecolor=ec, lw=1.3, height=0.55, zorder=3)
        ax.text(d + 0.35, i + 0.34, f'{d:+.1f} pp  {star(pval[m])}', va='center',
                ha='left', fontsize=9.5, fontweight='bold', color=ec, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=10.5, fontweight='bold')
    ax.set_xlim(xmin, xmax); ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlabel('Harness payoff: A4 (retrieval-plus-tool) minus A3 (tool agent), percentage points', fontsize=10)
    ax.set_title('Harness payoff: A4 (retrieval-plus-tool) minus A3 (tool agent) by base model',
                 fontsize=12, pad=10)
    ax.grid(axis='x', alpha=0.25, ls=':')
    for s in ('left', 'right', 'top', 'bottom'):
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(1.0); ax.spines[s].set_color('#444')
    plt.tight_layout()
    p1 = OUT/'Figure_5_harness_decision.png'; p2 = OUT/'Figure_5_harness_decision.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig 5 → {p1.name}')

# =================================================================
# (Sub-domain diverging heatmap is now panel b of the merged Fig 2.)
# Fig 3 — reliability bubble scatter (20 wrapper × base-model cells) + harness payoff
# =================================================================
def fig_4_reliability():
    hr_rows = [json.loads(l) for l in HR_FILE.read_text().splitlines()]
    hr_df = pd.DataFrame(hr_rows)
    _sub = set((ROOT / 'runs/v4_rerun/_hr/grounding_subsample_qids.txt').read_text().split())
    hr_df = hr_df[hr_df.qid.isin(_sub)]
    hr_df['arch'] = hr_df['arch'].map(DIR2ARCH)
    hr_df = hr_df.dropna(subset=['arch'])
    hr_df['model'] = hr_df['model'].map(KEY2MODEL)

    def safe_div(n, d): return n/d if d > 0 else 0
    # Strict hallucination rate: CONTRADICTED only (excludes NO_INFO).
    # NO_INFO is dominated by answer verbosity (terse GPT answers are mostly
    # not literal restatements of the reference quote and get labelled NO_INFO
    # even when factually fine), so it conflates verbosity with hallucination
    # across base models.
    HR = {m: {a:
        safe_div(
            hr_df[(hr_df.model==m)&(hr_df.arch==a)]['n_contradicted'].sum(),
            hr_df[(hr_df.model==m)&(hr_df.arch==a)]['n_claims'].sum())
        for a in ARCHS} for m in MODELS}
    HR_mean = {a: np.mean([HR[m][a] for m in MODELS]) for a in ARCHS}
    ACC_MA = {m: {a: LONG[(LONG.model==m)&(LONG.arch==a)]['score'].mean()
                  for a in ARCHS} for m in MODELS}
    IF_MA = {m: {a: (1 - LONG[(LONG.model==m)&(LONG.arch==a)]['parse_error'].mean())
                 for a in ARCHS} for m in MODELS}

    BASE_MARKER = {'GPT-5.4':'o', 'GPT-5.4-mini':'P', 'GPT-5.4-nano':'s', 'DeepSeek-V4':'D'}

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    # leave ~26 % on the right for legends
    plt.subplots_adjust(left=0.085, right=0.74, top=0.92, bottom=0.11)

    xs, ys, sizes = [], [], []
    for m in MODELS:
        for a in ARCHS:
            x, y, ifr = HR[m][a], ACC_MA[m][a], IF_MA[m][a]
            ax.scatter(x, y, s=80 + (ifr - 0.93) * 5000,
                       color=FILL[a], edgecolor=EDGE[a], lw=1.6,
                       marker=BASE_MARKER[m], alpha=0.92, zorder=5)
            xs.append(x); ys.append(y); sizes.append(ifr)

    # The cross-cell Pearson r is a base-model clustering artefact (Section 3.5),
    # so it is reported in the caption, not drawn as a trend line that would
    # assert exactly the pooled correlation the text disavows.
    r = np.corrcoef(xs, ys)[0, 1]

    ax.set_xlabel('Strict hallucination rate  (CONTRADICTED claims / total claims, per cell)',
                  fontsize=10.5)
    ax.set_ylabel('Overall accuracy  (per cell)', fontsize=11)
    ax.grid(alpha=0.22, ls=':'); _spines_box(ax)

    ax.set_xlim(min(xs) - 0.03, max(xs) + 0.03)
    ax.set_ylim(min(ys) - 0.04, max(ys) + 0.04)

    arch_h = [plt.Line2D([0],[0], marker='o', color='w',
                         markerfacecolor=FILL[a], markeredgecolor=EDGE[a],
                         markersize=11, markeredgewidth=1.6) for a in ARCHS]
    model_h = [plt.Line2D([0],[0], marker=BASE_MARKER[m], color='w',
                          markerfacecolor='#dddddd', markeredgecolor='#444',
                          markersize=11, markeredgewidth=1.4) for m in MODELS]
    leg1 = ax.legend(arch_h, [ARCH_DESC[a] for a in ARCHS],
                     loc='upper left', bbox_to_anchor=(1.02, 1.0),
                     fontsize=9.5, frameon=False,
                     title='Architecture (colour)', title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(model_h, MODELS,
              loc='upper left', bbox_to_anchor=(1.02, 0.52),
              fontsize=9.5, frameon=False,
              title='Base model (shape)', title_fontsize=10)
    fig.text(0.745, 0.16,
             'Marker size ∝ instruction-\nfollowing rate (0.94–1.00)',
             fontsize=8.5, color='#6c6c6c', style='italic')

    ax.set_title('Reliability bubble scatter  (20 wrapper × base-model cells)',
                 loc='left', fontweight='bold', fontsize=11.5, pad=8)

    p1 = OUT/'Figure_4_reliability.png'; p2 = OUT/'Figure_4_reliability.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig 4 → {p1.name}  (r={r:+.2f}, n={len(xs)})')
    return HR, HR_mean

# =================================================================
# Fig S1 — difficulty taxonomy, 3 models small multiples
# =================================================================
def fig_S1_taxonomy():
    SUBS = ['S1_exposure_factors','S2_microenv_conc','S3_trajectory_activity',
            'S4_dosimetry','S5_health']
    SUBS_L = ['S1','S2','S3','S4','S5']
    qtypes = ['true_false','calculation','open_ended']
    qt_L = {'true_false':'T/F','calculation':'Calc','open_ended':'Open'}
    diffs = ['easy','medium','hard']
    # color: k correct out of 5 archs — monochrome blue ramp (matches paper palette)
    k_colors = ['#eaeff5','#c5d4e6','#9bb4d4','#6e90bd','#456da6','#274a72']  # k=0..5, pale→dark
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.8), subplot_kw={'projection':'polar'})
    for col, m in enumerate(MODELS):
        ax = axes[col]
        sub = LONG[LONG.model==m]
        bins = []
        for s_i, s in enumerate(SUBS):
            for q_i, q in enumerate(qtypes):
                for d_i, d in enumerate(diffs):
                    cell = sub[(sub.subdomain==s)&(sub.question_type==q)&(sub.difficulty==d)]
                    if len(cell) == 0: continue
                    piv = cell.pivot_table(index='qid', columns='arch', values='score', aggfunc='first')
                    k_arr = (piv >= 0.5).sum(axis=1).values
                    bins.append((s_i, q_i, d_i, k_arr, len(cell)))
        N = len(bins)
        thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
        width = 2*np.pi / N
        for theta, (s_i, q_i, d_i, k_arr, n_q) in zip(thetas, bins):
            counts = np.bincount(k_arr, minlength=6).astype(float)
            counts = counts / counts.sum()  # frac of items per k bucket
            bottom = 0.0
            for k in range(6):
                if counts[k] > 0:
                    ax.bar(theta, counts[k], width=width*0.92, bottom=bottom,
                           color=k_colors[k], edgecolor='white', lw=0.4)
                    bottom += counts[k]
        # sub-domain labels at outer ring boundaries (3 qtypes × 3 diffs ≈ 9 wedges per SD)
        sd_starts = {}
        for theta, (s_i, *_) in zip(thetas, bins):
            sd_starts.setdefault(s_i, []).append(theta)
        for s_i, thetas_ in sd_starts.items():
            mid = (min(thetas_) + max(thetas_)) / 2
            ax.text(mid, 1.18, SUBS_L[s_i], ha='center', va='center',
                    fontsize=11, fontweight='bold')
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylim(0, 1.05)
        ax.spines['polar'].set_visible(False)
        ax.set_title(m, fontsize=12, fontweight='bold',
                     color=MODEL_COLOR[m], pad=18)
    legend_handles = [plt.Rectangle((0,0),1,1, fc=k_colors[k], ec='white') for k in range(6)]
    fig.legend(legend_handles, [f'k = {k}' for k in range(6)],
               loc='lower center', ncol=6, fontsize=10, frameon=False,
               title='Number of architectures (0–5) that answer the item correctly',
               title_fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Item-level taxonomy: 1,027 curated items binned by sub-domain × question type × difficulty',
                 fontsize=11.5, fontweight='bold', y=0.98)
    p1 = OUT/'Figure_S1_difficulty_taxonomy.png'; p2 = OUT/'Figure_S1_difficulty_taxonomy.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig S1 → {p1.name}')

# =================================================================
# Fig S2 — efficiency × robustness scatter
# =================================================================
def fig_S2_efficiency_robustness():
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    xs, ys, labels = [], [], []
    for m in MODELS:
        sub = LONG[LONG.model==m]
        for a in ARCHS:
            cell = sub[sub.arch==a]
            toks = (cell['in_tokens'] + cell['out_tokens']).sum()
            cor = cell['score'].sum()
            eff = toks/cor if cor > 0 else np.nan
            perr = cell['parse_error'].mean()
            xs.append(eff); ys.append(perr); labels.append((m, a))
    for (m, a), xv, yv in zip(labels, xs, ys):
        ax.scatter(xv, yv, s=270, color=FILL[a], edgecolor=MODEL_COLOR[m],
                   lw=2.0, marker=MARKER[a], zorder=4)
        ax.annotate(a, (xv, yv), xytext=(0, 0), textcoords='offset points',
                    ha='center', va='center', fontsize=7.5, fontweight='bold',
                    color=EDGE[a], zorder=5)
    # Two legends: shapes for arch, ring color for model
    h_archs = [plt.Line2D([0],[0], marker=MARKER[a], color='w',
                          markerfacecolor=FILL[a], markeredgecolor=EDGE[a],
                          markersize=10, markeredgewidth=1.5) for a in ARCHS]
    h_models = [plt.Line2D([0],[0], marker='o', color='w',
                           markerfacecolor='#eeeeee', markeredgecolor=MODEL_COLOR[m],
                           markersize=10, markeredgewidth=2.0) for m in MODELS]
    leg1 = ax.legend(h_archs, [ARCH_DESC[a] for a in ARCHS],
                     loc='upper left', fontsize=8.5, frameon=False,
                     title='Architecture (shape)', title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(h_models, MODELS, loc='upper right', fontsize=8.5, frameon=False,
              title='Base model (ring color)', title_fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('Tokens per correct answer (log scale)', fontsize=11)
    ax.set_ylabel('Parse-error rate', fontsize=11)
    ax.set_title('Efficiency × robustness frontier (20 wrapper × model cells)',
                 loc='left', fontweight='bold', fontsize=11.5)
    ax.grid(alpha=0.22, ls=':'); _spines_box(ax)
    ax.text(0.99, 0.02, 'Bottom-left = best (low cost, low failure rate)',
            transform=ax.transAxes, fontsize=8.5, color='#6c6c6c',
            ha='right', va='bottom', style='italic')

    p1 = OUT/'Figure_S2_efficiency_robustness.png'; p2 = OUT/'Figure_S2_efficiency_robustness.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig S2 → {p1.name}')

def fig_S3_evidence_use():
    """SI ablation: legacy plain retrieval versus the constrained main arms.

    Multiplicity is controlled across the eight displayed comparisons.  The
    plus labels are retained only to distinguish the ablation arms: A2+ and
    A4+ are the constrained variants renamed A2 and A4 in the main analysis.
    """
    raw = pd.read_parquet(PARQ)
    ABL_MODELS = ['GPT-5.4', 'GPT-5.4-mini', 'GPT-5.4-nano', 'DeepSeek-V4']
    # Plain and constrained variants retain exactly the same architecture
    # colour.  Marker shape and the A2/A2+ or A4/A4+ label carry variant
    # identity, preventing the SI from silently redefining the main palette.
    PAL = {'A2':(FILL['A2'], EDGE['A2']), 'A2+':(FILL['A2'], EDGE['A2']),
           'A4':(FILL['A4'], EDGE['A4']), 'A4+':(FILL['A4'], EDGE['A4'])}
    VARIANT_MARKER = {'A2':'o', 'A2+':'s', 'A4':'o', 'A4+':'s'}
    KEY = {'A2':'A2_rag', 'A2+':'A2p_rag_constrained',
           'A4':'A4_hybrid', 'A4+':'A4p_hybrid_constrained'}
    def acc(m, a):
        return raw[(raw.model==m)&(raw.arch==KEY[a])].score.mean()
    def paired_p(m, plain, con):
        piv = raw[raw.model==m].pivot_table(index='qid', columns='arch', values='score')
        pair = piv[[KEY[plain], KEY[con]]].dropna()
        try:
            return wilcoxon(pair[KEY[plain]], pair[KEY[con]]).pvalue
        except Exception:
            return 1.0
    comparisons = [(MODEL_KEY[m], plain, con)
                   for m in ABL_MODELS for plain, con in [('A2','A2+'), ('A4','A4+')]]
    adjusted = _holm_adjust([paired_p(*comparison) for comparison in comparisons])
    p_holm = {comparison: p for comparison, p in zip(comparisons, adjusted)}
    fig, ax = plt.subplots(figsize=(11, 7.5))
    rowsep, off, yticks = 1.0, 0.25, []
    for i, m in enumerate(ABL_MODELS):
        mk = MODEL_KEY[m]
        y1, y2 = i*rowsep + off, i*rowsep - off
        yticks.append(i*rowsep)
        for (plain, con, y) in [('A2','A2+',y1), ('A4','A4+',y2)]:
            xp, xc = acc(mk, plain), acc(mk, con)
            ax.plot([xp, xc], [y, y], color='#bbbbbb', lw=3.5, alpha=0.55, zorder=2)
            ax.scatter([xp], [y], s=220, c=PAL[plain][0], edgecolor=PAL[plain][1],
                       marker=VARIANT_MARKER[plain], lw=1.4, zorder=3)
            ax.scatter([xc], [y], s=220, c=PAL[con][0], edgecolor=PAL[con][1],
                       marker=VARIANT_MARKER[con], lw=1.4, zorder=3)
            ax.text(min(xp,xc)-0.004, y, plain, ha='right', va='center', fontsize=9, fontweight='bold', color=PAL[plain][1])
            ax.text(max(xp,xc)+0.004, y, con, ha='left', va='center', fontsize=9, fontweight='bold', color=PAL[con][1])
            dy = 0.16 if y == y1 else -0.22
            ax.annotate(f'{(xc-xp)*100:+.1f} pp  {_p_label(p_holm[(mk,plain,con)])}',
                        xy=((xp+xc)/2, y+dy), ha='center', fontsize=9.5,
                        color=PAL[con][1], fontweight='bold')
    ax.set_yticks(yticks); ax.set_yticklabels(ABL_MODELS, fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', length=4, pad=8)
    ax.set_ylim(-0.7, len(ABL_MODELS) - 1 + 0.7)
    for i in range(len(ABL_MODELS)-1):
        ax.axhline(i*rowsep + rowsep/2, color='#e6e6e6', lw=1.0)
    ax.set_xlabel('Overall accuracy', fontsize=11)
    ax.set_title('Evidence-use prompt ablation: legacy plain arm versus constrained main arm\n'
                 'paired Wilcoxon p values, Holm-adjusted across all eight comparisons',
                 fontsize=11.5, pad=10)
    ax.grid(axis='x', alpha=0.25); _spines_box(ax)
    plt.subplots_adjust(top=0.85, bottom=0.13, left=0.17, right=0.97)
    p1 = OUT/'Figure_S3_evidence_use.png'; p2 = OUT/'Figure_S3_evidence_use.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig S3 → {p1.name}')

def _wilson(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion, returned in percent (p, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * (p*(1-p)/n + z*z/(4*n*n))**0.5 / d
    return p*100, max(0.0, (c-h)*100), (c+h)*100

def fig_S4_mechanism():
    """SI: A4 sub-additivity is a control-flow collapse on weak base models.
    Dumbbell A3 -> A4 per base model with Wilson 95% CIs: (a) open-ended
    answer-type collapse, (b) calculation non-numeric rate. Colours match the
    main figures (A3 gold, A4 rose); a red connector flags a >10 pp jump."""
    summ = json.loads((ROOT / 'runs/v4_rerun/_mechanism/summary.json').read_text())
    order = ['gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'deepseek-v4']
    ys = list(range(len(order)))[::-1]        # GPT-5.4 at the top row
    A3C, A3E = FILL['A3'], EDGE['A3']          # tool agent: gold (as in Fig 2, 3)
    A4C, A4E = FILL['A4'], EDGE['A4']          # harness: rose
    RED = ALERT
    XMAX = 82
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3))
    plt.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.19, wspace=0.30)
    panels = [('open', 'a', 'Prose question answered as boolean / number / <15 chars (%)'),
              ('calc', 'b', 'Calculation answered with no parseable number (%)')]
    for ax, (grp, title, sub) in zip(axes, panels):
        # column direction headers (A3 on the left, A4 to the right)
        ax.text(8, 3.62, 'A3\ntool agent', ha='center', va='bottom', fontsize=8.2,
                color=A3E, fontweight='bold', linespacing=1.1)
        ax.text(46, 3.62, 'A4\nharness (retrieved tool)', ha='center', va='bottom', fontsize=8.2,
                color=A4E, fontweight='bold', linespacing=1.1)
        for m, y in zip(order, ys):
            n = summ[m][f'{grp}_n']
            p3, lo3, hi3 = _wilson(summ[m][f'{grp}_k_A3'], n)
            p4, lo4, hi4 = _wilson(summ[m][f'{grp}_k_A4'], n)
            delta = p4 - p3; sig = delta > 10
            ax.plot([lo3, hi3], [y, y], '-', color=A3E, lw=0.9, alpha=0.45, zorder=1)
            ax.plot([lo4, hi4], [y, y], '-', color=A4E, lw=0.9, alpha=0.45, zorder=1)
            ax.plot([p3, p4], [y, y], '-', color=RED if sig else '#c4c4c4',
                    lw=3.4 if sig else 1.8, zorder=2, solid_capstyle='round')
            ax.scatter([p3], [y], s=70, c=A3C, edgecolor=A3E, lw=1.2, zorder=4)
            ax.scatter([p4], [y], s=70, c=A4C, edgecolor=A4E, lw=1.2, zorder=4)
            ax.text(p3-2.2, y, f'{p3:.0f}', ha='right', va='center', fontsize=8.5,
                    color=A3E, fontweight='bold')
            ax.text(p4+2.2, y, f'{p4:.0f}', ha='left', va='center', fontsize=8.5,
                    color=RED if sig else A4E, fontweight='bold')
            ax.text(XMAX, y, f'{delta:+.0f} pp', ha='right', va='center', fontsize=8.5,
                    color=RED if sig else '#9a9a9a', fontweight='bold' if sig else 'normal')
        ax.set_yticks(ys); ax.set_yticklabels([KEY2MODEL[m] for m in order], fontsize=9.5)
        for tl, m in zip(ax.get_yticklabels(), order):
            if m == 'gpt-5.4-nano':
                tl.set_color(RED); tl.set_fontweight('bold')
        ax.set_xlim(-7, XMAX+2); ax.set_ylim(-0.6, 4.15)
        ax.set_xticks([0, 20, 40, 60, 80]); ax.set_xlabel(sub, fontsize=9)
        ax.tick_params(length=3); ax.grid(axis='x', alpha=0.18); _spines_box(ax)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=8)
    leg_h = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=A3C, markeredgecolor=A3E,
                        markersize=9, markeredgewidth=1.2),
             plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=A4C, markeredgecolor=A4E,
                        markersize=9, markeredgewidth=1.2),
             plt.Line2D([0],[0], color='#c4c4c4', lw=2.0),
             plt.Line2D([0],[0], color=RED, lw=3.2)]
    fig.legend(leg_h, ['A3: tool agent', 'A4: harness (retrieved tool)', 'change A3 → A4',
                       'red: > +10 pp degradation'],
               loc='lower center', ncol=4, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 0.01),
               columnspacing=2.0, handletextpad=0.6)
    p1 = OUT/'Figure_3_mechanism.png'; p2 = OUT/'Figure_3_mechanism.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig 3 (mechanism) → {p1.name}')


def _extractable_number(value):
    """Return True when an answer field contains a finite scalar number."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    text = str(value).strip().replace(',', '')
    match = re.search(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', text)
    return bool(match)


def _if_pass(row, question_type):
    """Type-specific instruction-following rule stated in manuscript §2.5."""
    if row.get('parse_error'):
        return False
    answer = row.get('answer')
    if question_type == 'true_false':
        return isinstance(answer, bool) or str(answer).strip().lower() in {'true', 'false'}
    if question_type == 'calculation':
        unit = str(row.get('unit') or '').strip().lower()
        return _extractable_number(answer) and unit not in {'', 'none', 'null'}
    return answer is not None and bool(str(answer).strip())


def _load_instruction_following():
    """Load the original 1,004 and programmatic 100 streams from raw trajectories."""
    qtype = LONG.drop_duplicates('qid').set_index('qid')['question_type'].to_dict()
    result = {}
    for model in MODELS:
        model_key = MODEL_KEY[model]
        result[model] = {}
        for raw_arch, arch in DIR2ARCH.items():
            records = {}
            for run_root in (RUNS_DIR, TOOL_RUNS_DIR):
                path = run_root / model_key / raw_arch / 'run_1.jsonl'
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    records[row.get('qid')] = row
            flags = [_if_pass(row, qtype.get(qid)) for qid, row in records.items() if qid in qtype]
            result[model][arch] = (float(np.mean(flags)), len(flags)) if flags else (np.nan, 0)
    return result


def _wilson_ci(rate, n, z=1.96):
    if n <= 0 or np.isnan(rate):
        return (np.nan, np.nan)
    den = 1 + z*z/n
    centre = (rate + z*z/(2*n)) / den
    half = z * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / den
    return centre-half, centre+half


def _ratio_ci(frame, numerator, denominator, seed, n_boot=1000):
    num = numerator(frame).to_numpy(dtype=float)
    den = denominator(frame).to_numpy(dtype=float)
    point = num.sum() / den.sum() if den.sum() else np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(frame), size=(n_boot, len(frame)))
    boot_num = num[idx].sum(axis=1)
    boot_den = den[idx].sum(axis=1)
    boot = np.divide(boot_num, boot_den, out=np.full(n_boot, np.nan), where=boot_den > 0)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def fig_4_reliability_diagnostics():
    """Reliability metrics with explicit denominators and a clear visual hierarchy."""
    hr_df = pd.DataFrame(json.loads(line) for line in HR_FILE.read_text().splitlines())
    _sub = set((ROOT / 'runs/v4_rerun/_hr/grounding_subsample_qids.txt').read_text().split())
    hr_df = hr_df[hr_df.qid.isin(_sub)]
    hr_df['arch'] = hr_df['arch'].map(DIR2ARCH)
    hr_df = hr_df.dropna(subset=['arch'])
    hr_df['model'] = hr_df['model'].map(KEY2MODEL)
    if_data = _load_instruction_following()

    summaries = {'if': {}, 'coverage': {}, 'contradiction': {}}
    for mi, model in enumerate(MODELS):
        for ai, arch in enumerate(ARCHS):
            rate, n = if_data[model][arch]
            lo, hi = _wilson_ci(rate, n)
            summaries['if'][(model, arch)] = (rate, lo, hi, n)
            sub = hr_df[(hr_df.model == model) & (hr_df.arch == arch)]
            summaries['coverage'][(model, arch)] = (*_ratio_ci(
                sub,
                lambda x: x['n_supported'] + x['n_contradicted'],
                lambda x: x['n_claims'],
                seed=100 + mi*10 + ai), len(sub))
            summaries['contradiction'][(model, arch)] = (*_ratio_ci(
                sub,
                lambda x: x['n_contradicted'],
                lambda x: x['n_supported'] + x['n_contradicted'],
                seed=200 + mi*10 + ai), len(sub))

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.6), sharex=True)
    metrics = [
        ('if', 'Instruction following ↑', 'Instruction-following rate', (0.68, 1.01)),
        ('coverage', 'Adjudication coverage ↑', 'Adjudicated claims', (0, 0.55)),
        ('contradiction', 'Contradiction rate ↓', 'Contradiction among\nadjudicated claims', (0, 0.20)),
    ]
    model_marker = {'GPT-5.4':'o', 'GPT-5.4-mini':'P', 'GPT-5.4-nano':'s', 'DeepSeek-V4':'D'}
    fig4_color = {'GPT-5.4':'#0b4fa3', 'GPT-5.4-mini':'#078a8f',
                  'GPT-5.4-nano':'#74409a', 'DeepSeek-V4':'#d58d00'}
    offsets = np.linspace(-0.13, 0.13, len(MODELS))
    xbase = np.arange(len(ARCHS))
    for pi, (key, title, ylabel, ylim) in enumerate(metrics):
        ax = axes[pi]
        # A3 is the focal configuration in the manuscript; the neutral band is
        # an attentional cue, not a claim of a statistically defined optimum.
        ax.axvspan(2.77, 3.23, color='#dce8f5', alpha=0.52, lw=0, zorder=0)
        ax.text(3, 1.018, 'Focal A3', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8.2, color='#0b4fa3',
                fontweight='bold', style='italic')
        for mi, model in enumerate(MODELS):
            vals, los, his = [], [], []
            for arch in ARCHS:
                v, lo, hi, _ = summaries[key][(model, arch)]
                vals.append(v); los.append(lo); his.append(hi)
            vals = np.asarray(vals); los = np.asarray(los); his = np.asarray(his)
            xx = xbase + offsets[mi]
            if key == 'if':
                ax.plot(xx, vals, color=fig4_color[model], lw=1.9,
                        alpha=0.88, zorder=2)
            ax.errorbar(xx, vals,
                        yerr=[np.maximum(vals-los, 0), np.maximum(his-vals, 0)],
                        fmt=model_marker[model],
                        color=fig4_color[model], markerfacecolor='white', markeredgewidth=1.55,
                        ecolor=mpl.colors.to_rgba(fig4_color[model], 0.42),
                        ms=7.0, elinewidth=0.9, capthick=0.9, capsize=2.0,
                        lw=0, zorder=4, label=model)
        ax.set_xticks(xbase); ax.set_xticklabels(ARCHS, fontsize=9.5)
        for tick, arch in zip(ax.get_xticklabels(), ARCHS):
            if arch == 'A3':
                tick.set_color('#0b4fa3'); tick.set_fontweight('bold')
        ax.set_ylabel(ylabel, fontsize=9.8)
        ax.set_ylim(*ylim)
        if key == 'if':
            ax.set_yticks(np.arange(0.70, 1.001, 0.05))
        elif key == 'coverage':
            ax.set_yticks(np.arange(0, 0.51, 0.10))
        else:
            ax.set_yticks(np.arange(0, 0.201, 0.05))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis='y', color='#9aa5b1', alpha=0.22, lw=0.8, ls='--')
        ax.grid(axis='x', visible=False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#aeb6bf'); ax.spines[side].set_linewidth(0.8)
        ax.set_title(f"{chr(ord('a')+pi)}    {title}", loc='left',
                     fontweight='bold', fontsize=12.5, pad=14)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=model_marker[m], linestyle='none',
                      markerfacecolor='white', markeredgecolor=fig4_color[m],
                      markeredgewidth=1.6, markersize=7.5, label=m) for m in MODELS]
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                        ncol=4, frameon=True, fontsize=9.4, columnspacing=2.2,
                        handletextpad=0.6, borderpad=0.7)
    legend.get_frame().set_edgecolor('#d3d8de')
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_facecolor('#ffffff')
    fig.supxlabel('Architecture', fontsize=10.5, fontweight='bold', y=0.055)
    fig.subplots_adjust(left=0.064, right=0.988, top=0.78, bottom=0.17, wspace=0.30)
    p1 = OUT/'Figure_4_reliability_diagnostics.png'; p2 = OUT/'Figure_4_reliability_diagnostics.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig 4 (reliability diagnostics) → {p1.name}')


def _holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _p_label(p):
    if p < 1e-4:
        return 'p_Holm < 10⁻⁴'
    if p < 0.01:
        return f'p_Holm = {p:.3f}'
    return f'p_Holm = {p:.2f}'


def fig_5_configuration_contrast():
    """A4-minus-A3 as a complete-configuration contrast with heterogeneity panels."""
    pair = LONG[LONG.arch.isin(['A3', 'A4'])].pivot_table(
        index=['model', 'qid', 'question_type'], columns='arch', values='score', aggfunc='first')
    pair = pair.dropna(subset=['A3', 'A4']).reset_index()
    pair['delta_pp'] = (pair['A4'] - pair['A3']) * 100
    pair['stream'] = np.where(pair['qid'].str.startswith('TS'), 'Programmatic 100', 'Original 1,004')

    order = ['GPT-5.4', 'GPT-5.4-mini', 'DeepSeek-V4', 'GPT-5.4-nano']
    rng = np.random.default_rng(42)
    effects, pvalues = [], []
    for model in order:
        values = pair.loc[pair.model == model, 'delta_pp'].to_numpy()
        boot_idx = rng.integers(0, len(values), size=(5000, len(values)))
        boot = values[boot_idx].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        p = wilcoxon(values, zero_method='wilcox').pvalue
        effects.append((values.mean(), float(lo), float(hi), len(values)))
        pvalues.append(p)
    p_holm = _holm_adjust(pvalues)

    qtypes = ['true_false', 'calculation', 'open_ended']
    qlabels = ['True/false', 'Calculation', 'Open-ended']
    streams = ['Original 1,004', 'Programmatic 100']
    qmat = np.array([[pair[(pair.model==m)&(pair.question_type==q)].delta_pp.mean()
                      for q in qtypes] for m in order])
    smat = np.array([[pair[(pair.model==m)&(pair.stream==s)].delta_pp.mean()
                      for s in streams] for m in order])

    fig = plt.figure(figsize=(15.2, 5.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.38, 1.02, 0.78], wspace=0.33,
                          left=0.075, right=0.95, top=0.88, bottom=0.20)
    axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))

    y = np.arange(len(order))
    axa.axvline(0, color='#444', lw=1.1, zorder=1)
    for i, model in enumerate(order):
        mean, lo, hi, n = effects[i]
        axa.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o', ms=7.5,
                     color=MODEL_COLOR[model], markerfacecolor='white', markeredgewidth=1.6,
                     elinewidth=1.5, capsize=3, zorder=3)
        axa.text(6.0, i, f'{mean:+.1f} pp  [{lo:+.1f}, {hi:+.1f}]\n{_p_label(p_holm[i])}; n={n}',
                 ha='left', va='center', fontsize=8.2, color='#444')
    axa.set_yticks(y); axa.set_yticklabels(order, fontsize=9.5, fontweight='bold')
    axa.invert_yaxis(); axa.set_xlim(-31, 23)
    axa.set_xlabel('A4 − A3 mean accuracy difference (pp)\n95% paired item-bootstrap CI', fontsize=9.5)
    axa.grid(axis='x', alpha=0.22, ls=':'); _spines_box(axa)
    axa.set_title('a  Overall complete-configuration contrast', loc='left', fontweight='bold', fontsize=11, pad=8)
    axa.text(0.01, -0.17, 'DeepSeek-V4 main-run difference is not stable across the four-seed replication.',
             transform=axa.transAxes, fontsize=7.4, color='#777', style='italic', ha='left')

    vmax = 50
    im_b = axb.imshow(qmat, cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(qmat.shape[0]):
        for j in range(qmat.shape[1]):
            axb.text(j, i, f'{qmat[i,j]:+.1f}', ha='center', va='center', fontsize=9,
                     color='white' if abs(qmat[i,j]) > 22 else '#333', fontweight='bold')
    axb.set_xticks(range(3)); axb.set_xticklabels(qlabels, rotation=28, ha='right', fontsize=8.7)
    axb.set_yticks(range(4)); axb.set_yticklabels(order, fontsize=8.5)
    axb.set_title('b  Heterogeneity by question type (pp)', loc='left', fontweight='bold', fontsize=11, pad=8)
    _spines_box(axb)

    axc.imshow(smat, cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(smat.shape[0]):
        for j in range(smat.shape[1]):
            axc.text(j, i, f'{smat[i,j]:+.1f}', ha='center', va='center', fontsize=9,
                     color='white' if abs(smat[i,j]) > 22 else '#333', fontweight='bold')
    axc.set_xticks(range(2)); axc.set_xticklabels(['LLM-guided\n927', 'Programmatic\n100'], fontsize=8.7)
    axc.set_yticks(range(4)); axc.set_yticklabels(order, fontsize=8.5)
    axc.set_title('c  Heterogeneity by data stream (pp)', loc='left', fontweight='bold', fontsize=11, pad=8)
    _spines_box(axc)
    cbar_ax = fig.add_axes([0.962, 0.25, 0.012, 0.53])
    cb = fig.colorbar(im_b, cax=cbar_ax)
    cb.set_label('A4 − A3 (pp)', fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
    fig.suptitle('A4 versus A3: observed configuration differences, not an isolated retrieval effect',
                 fontsize=12.5, y=0.975)
    p1 = OUT/'Figure_5_configuration_contrast.png'; p2 = OUT/'Figure_5_configuration_contrast.svg'
    plt.savefig(p1, dpi=180, bbox_inches='tight'); plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f'  Fig 5 (configuration contrast) → {p1.name}')


if __name__ == '__main__':
    # The manuscript editions of Figures 2 and 4 use compact letter-coded
    # benchmark markers and are maintained as dependency-light generators.
    from redesign_fig2 import draw as draw_fig2_logo
    from redesign_fig4 import draw as draw_fig4_logo

    print(f'Loaded {len(LONG)} rows (5 archs × {len(MODELS)} models × 1104 items)')
    fig_1_arch_mechanism()
    draw_fig2_logo()            # merged heatmaps + letter-coded global Pareto panel
    draw_fig4_logo()            # letter-coded reliability diagnostics
    fig_5_configuration_contrast()
    # fig_S3_evidence_use()  # original-campaign-only arms; replaced by factorial figure
    fig_S1_taxonomy()
    fig_S2_efficiency_robustness()
    # fig_S4_mechanism()  # old collapse dumbbell (Fig 3); replaced by environment-effect exhibit
    print(f'All saved → {OUT}')
