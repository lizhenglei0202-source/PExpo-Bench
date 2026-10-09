#!/usr/bin/env python3
"""Agreement analysis for the judge-calibration study.

Reads per_row_double_judge.jsonl (output of run_double_judge.py) and, if the
human columns in human_rating_sheet.csv have been filled, computes
judge-human agreement as well. No API calls.

Reports:
  - per-judge score means (0-5) overall and per source model
  - judge-judge: Pearson r, Spearman rho, exact agreement, |diff|<=1,
    Cohen's kappa (unweighted, linear, quadratic)
  - human: rater1-rater2 agreement, and each judge vs mean-human score
    (quadratic-weighted kappa uses rounded human means)

Usage: python analyze_agreement.py [--judged per_row_double_judge.jsonl]
                                    [--sheet human_rating_sheet.csv]
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import pandas as pd

PKG = pathlib.Path(__file__).resolve().parent


def cohen_kappa(a: np.ndarray, b: np.ndarray, weights: str | None = None,
                n_cat: int = 6) -> float:
    """Cohen's kappa for integer labels 0..n_cat-1.

    weights: None (unweighted), 'linear', or 'quadratic'.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    conf = np.zeros((n_cat, n_cat))
    for i, j in zip(a, b):
        conf[i, j] += 1
    n = conf.sum()
    idx = np.arange(n_cat)
    if weights is None:
        w = 1.0 - np.eye(n_cat)
    elif weights == 'linear':
        w = np.abs(idx[:, None] - idx[None, :]) / (n_cat - 1)
    elif weights == 'quadratic':
        w = ((idx[:, None] - idx[None, :]) / (n_cat - 1)) ** 2
    else:
        raise ValueError(weights)
    row = conf.sum(axis=1)
    col = conf.sum(axis=0)
    expected = np.outer(row, col) / n
    po = (w * conf).sum() / n
    pe = (w * expected).sum() / n
    if pe == 0:
        return 1.0
    return 1.0 - po / pe


def pair_stats(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ai = np.rint(a).astype(int).clip(0, 5)
    bi = np.rint(b).astype(int).clip(0, 5)
    sa = pd.Series(a).rank()
    sb = pd.Series(b).rank()
    return {
        'pair': f'{name_a} vs {name_b}',
        'n': int(len(a)),
        'pearson_r': round(float(np.corrcoef(a, b)[0, 1]), 4),
        'spearman_rho': round(float(np.corrcoef(sa, sb)[0, 1]), 4),
        'exact_agree': round(float((ai == bi).mean()), 4),
        'within_1': round(float((np.abs(ai - bi) <= 1).mean()), 4),
        'kappa_unweighted': round(cohen_kappa(ai, bi, None), 4),
        'kappa_linear': round(cohen_kappa(ai, bi, 'linear'), 4),
        'kappa_quadratic': round(cohen_kappa(ai, bi, 'quadratic'), 4),
        'mean_diff': round(float((a - b).mean()), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--judged', default=str(PKG / 'per_row_double_judge.jsonl'))
    p.add_argument('--sheet', default=str(PKG / 'human_rating_sheet.csv'))
    p.add_argument('--out', default=str(PKG / 'agreement_report.json'))
    args = p.parse_args()

    report: dict = {}

    # ---------- judge scores ----------
    jf = pathlib.Path(args.judged)
    if not jf.exists():
        print(f'[warn] {jf} not found - run run_double_judge.py first')
        wide = None
    else:
        rows = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
        jd = pd.DataFrame(rows)
        jd = jd[jd.score_0_5 >= 0]
        wide = jd.pivot_table(index=['sample_id', 'model'], columns='judge',
                              values='score_0_5', aggfunc='last').reset_index()
        judges = [c for c in wide.columns if c not in ('sample_id', 'model')]
        report['n_judged_rows'] = int(len(wide))
        report['per_judge_mean'] = {
            j: round(float(wide[j].mean()), 4) for j in judges}
        report['per_judge_mean_by_model'] = {
            j: {m: round(float(g[j].mean()), 4)
                for m, g in wide.groupby('model')} for j in judges}
        if len(judges) == 2:
            both = wide.dropna(subset=judges)
            report['judge_judge'] = pair_stats(
                both[judges[0]].values, both[judges[1]].values,
                judges[0], judges[1])

    # ---------- human ratings ----------
    hs = pd.read_csv(args.sheet)
    for c in ('rater1_score_0_5', 'rater2_score_0_5'):
        hs[c] = pd.to_numeric(hs[c], errors='coerce')
    filled = hs.dropna(subset=['rater1_score_0_5', 'rater2_score_0_5'])
    report['n_human_rated'] = int(len(filled))
    if len(filled) >= 10:
        report['human_human'] = pair_stats(
            filled['rater1_score_0_5'].values,
            filled['rater2_score_0_5'].values, 'rater1', 'rater2')
        filled = filled.assign(
            human_mean=(filled['rater1_score_0_5']
                        + filled['rater2_score_0_5']) / 2)
        if wide is not None:
            merged = wide.merge(
                filled[['sample_id', 'human_mean']], on='sample_id')
            judges = [c for c in wide.columns
                      if c not in ('sample_id', 'model')]
            report['judge_human'] = {
                j: pair_stats(merged.dropna(subset=[j])[j].values,
                              merged.dropna(subset=[j])['human_mean'].values,
                              j, 'human_mean')
                for j in judges}
    else:
        report['human_note'] = ('human columns not (sufficiently) filled - '
                                'judge-human agreement skipped')

    print(json.dumps(report, indent=2))
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    print(f'[saved] {args.out}')
    return 0


if __name__ == '__main__':
    sys_exit = main()
    raise SystemExit(sys_exit)
