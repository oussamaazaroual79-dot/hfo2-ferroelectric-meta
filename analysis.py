"""Statistical analysis of the HfO2 ferroelectric literature database (hfo2_db_v4.csv)."""
import json, numpy as np, pandas as pd
from scipy import stats

rng = np.random.default_rng(42)
db = pd.read_csv('hfo2_db_v4.csv')
db['physically_implausible'] = db.physically_implausible.astype(bool)
db['unsaturated'] = db.unsaturated.fillna(False).astype(bool)
d = db[~db.physically_implausible & ~db.unsaturated].copy()

dep_map = {'ALD': 'ALD', 'TALD': 'ALD', 'PEALD': 'ALD', 'PAALD': 'ALD', 'PLD': 'PLD', 'CSD': 'CSD', 'Sputtering': 'Sputtering'}
d['dep'] = d.deposition.map(dep_map)
def fam(x):
    if x in ('Zr', 'Zr+La interlayers'): return 'Zr (HZO)'
    if x in ('La', 'Si', 'Y', 'Al'): return x
    return 'Other'
d['dopfam'] = d.dopant.map(fam)
d['epi'] = (d.crystal == 'epitaxial').astype(int)
d['study'] = d.source.str.extract(r'^([^,]+, [^,]+,? [^,]*?\d{4})')[0].fillna(d.source)
d['study'] = d.source.str.split('(').str[0].str.strip()
d['logt'] = np.log10(d.thickness_nm)
d['logEc'] = np.log10(d.Ec_MV_cm)
d['figure_digitized'] = d.method.str.contains('figure', na=False)
R = {}

# ---------- 1. Descriptives ----------
def q(s): s = s.dropna(); return [round(s.median(), 2), round(s.quantile(.25), 2), round(s.quantile(.75), 2), int(len(s))]
R['n_total'] = int(len(db)); R['n_usable'] = int(len(d)); R['n_implausible'] = int(db.physically_implausible.sum()); R['n_unsaturated'] = int(db.unsaturated.sum())
R['n_studies'] = int(db.source.nunique()); R['n_with_Ec'] = int(d.Ec_MV_cm.notna().sum())
R['n_figure_digitized'] = int(d.figure_digitized.sum())
R['by_crystal'] = {c: {'Pr': q(g.Pr_uC_cm2), 'Ec': q(g.Ec_MV_cm), 't': q(g.thickness_nm)} for c, g in d.groupby('crystal')}
R['by_dopant'] = {c: {'Pr': q(g.Pr_uC_cm2), 'Ec': q(g.Ec_MV_cm), 'n_studies': int(g.study.nunique())} for c, g in d.groupby('dopfam')}
R['by_dep'] = {c: {'Pr': q(g.Pr_uC_cm2), 'Ec': q(g.Ec_MV_cm)} for c, g in d.groupby('dep')}
e, p = d[d.epi == 1], d[d.epi == 0]
R['MW_Ec'] = float(stats.mannwhitneyu(e.Ec_MV_cm.dropna(), p.Ec_MV_cm.dropna()).pvalue)
R['MW_Pr'] = float(stats.mannwhitneyu(e.Pr_uC_cm2.dropna(), p.Pr_uC_cm2.dropna()).pvalue)
R['KW_Pr_dopant_poly'] = float(stats.kruskal(*[g.Pr_uC_cm2.dropna() for _, g in p.groupby('dopfam') if len(g) >= 5]).pvalue)
R['KW_Ec_dopant_poly'] = float(stats.kruskal(*[g.Ec_MV_cm.dropna() for _, g in p.groupby('dopfam') if g.Ec_MV_cm.notna().sum() >= 5]).pvalue)

# ---------- 2. Ec-thickness scaling: pooled and within-series ----------
def powfit(x, y):
    m = np.isfinite(np.log(x)) & np.isfinite(np.log(y)); x, y = np.log10(x[m]), np.log10(y[m])
    s, i, r, pv, se = stats.linregress(x, y); return s, se, r**2, pv, int(m.sum())
R['Ec_t_pooled'] = {}
for c, g in d.dropna(subset=['thickness_nm', 'Ec_MV_cm']).groupby('crystal'):
    s, se, r2, pv, n = powfit(g.thickness_nm.values, g.Ec_MV_cm.values)
    rho = stats.spearmanr(g.thickness_nm, g.Ec_MV_cm)
    R['Ec_t_pooled'][c] = dict(slope=round(s, 3), se=round(se, 3), r2=round(r2, 3), p=float(pv), n=n, rho=round(rho.statistic, 3), rho_p=float(rho.pvalue))

# a "series" = films from one study sharing everything except the varied parameter
d['series_key'] = d.study + '|' + d['stack'].fillna('') + '|' + d.method.fillna('') + '|' + d.anneal_C.map(str) + '|' + d.dopant_conc.map(str) + '|' + d.architecture.fillna('')
series_t = []
for k, g in d.dropna(subset=['thickness_nm', 'Ec_MV_cm']).groupby('series_key'):
    if g.thickness_nm.nunique() >= 3 and not g.method.str.contains('review table', na=False).any():
        s, se, r2, pv, n = powfit(g.thickness_nm.values, g.Ec_MV_cm.values)
        series_t.append(dict(series=k, crystal=g.crystal.iloc[0], dopant=g.dopfam.iloc[0], n=n, tmin=g.thickness_nm.min(), tmax=g.thickness_nm.max(), slope=round(s, 3), se=round(se, 3)))
R['Ec_t_series'] = series_t

# ---------- 2b. Within-series (fixed-effects) Ec-t scaling vs pooled ----------
ser = d.dropna(subset=['thickness_nm', 'Ec_MV_cm'])
ser = ser[~ser.method.str.contains('review table', na=False)]
def fe_slope(df):
    x = df.logt - df.groupby('series_key').logt.transform('mean'); y = df.logEc - df.groupby('series_key').logEc.transform('mean')
    return float((x * y).sum() / (x * x).sum())
def fe_block(g, minpts=3):
    g = g[g.groupby('series_key').thickness_nm.transform('nunique') >= minpts]
    keys = g.series_key.unique()
    if len(keys) < 2: return None, None
    b = fe_slope(g)
    boots = np.array([fe_slope(pd.concat([g[g.series_key == k].assign(series_key=f'{k}#{i}') for i, k in enumerate(rng.choice(keys, len(keys)))])) for _ in range(2000)])
    return dict(slope=round(b, 3), ci=[round(np.percentile(boots, 2.5), 3), round(np.percentile(boots, 97.5), 3)], n_series=int(len(keys)), n_films=int(len(g)),
                share_boot_below_minus_half=round(float((boots < -0.5).mean()), 3), contains_minus_half=bool(np.percentile(boots, 2.5) <= -0.5 <= np.percentile(boots, 97.5)),
                contains_minus_two_thirds=bool(np.percentile(boots, 2.5) <= -2/3 <= np.percentile(boots, 97.5))), boots
R['FE_scaling'] = {}; BOOT = {}
for c in ['epitaxial', 'polycrystalline']:
    for lab, lo, hi, mp in [('t<=50nm', 0, 50, 3), ('all t', 0, 1e9, 3), ('t>=30nm', 30, 1e9, 2)]:
        res, bt = fe_block(ser[(ser.crystal == c) & (ser.thickness_nm >= lo) & (ser.thickness_nm <= hi)], mp)
        if res: R['FE_scaling'][f'{c} {lab}'] = res; BOOT[f'{c} {lab}'] = bt
if 'epitaxial t<=50nm' in BOOT and 'polycrystalline t<=50nm' in BOOT:
    diff = BOOT['epitaxial t<=50nm'] - BOOT['polycrystalline t<=50nm']
    R['FE_slope_diff_epi_minus_poly_thin'] = dict(mean=round(float(diff.mean()), 3), ci=[round(np.percentile(diff, 2.5), 3), round(np.percentile(diff, 97.5), 3)])
# HZO-only (same composition) comparison
hz = ser[ser.dopfam == 'Zr (HZO)']
R['FE_scaling_HZO'] = {c: fe_block(hz[(hz.crystal == c) & (hz.thickness_nm <= 50)])[0] for c in ['epitaxial', 'polycrystalline']}
# Ec at 10 nm from each series' own power-law fit (series spanning 10 nm)
off = []
for k, g in ser.groupby('series_key'):
    if g.thickness_nm.nunique() >= 3 and g.thickness_nm.min() <= 10 <= g.thickness_nm.max():
        b = np.polyfit(g.logt, g.logEc, 1); off.append(dict(series=k, crystal=g.crystal.iloc[0], Ec10=round(10**np.polyval(b, 1.0), 2)))
R['Ec_at_10nm_series'] = off
R['Ec_at_10nm_by_class'] = {c: [round(float(np.median([o['Ec10'] for o in off if o['crystal'] == c])), 2), sum(o['crystal'] == c for o in off)] for c in ['epitaxial', 'polycrystalline']}

# thickness-only series for Pr: thickness at maximum Pr
pk = []
for k, g in d.dropna(subset=['thickness_nm', 'Pr_uC_cm2']).groupby('series_key'):
    if g.thickness_nm.nunique() >= 4:
        g = g.sort_values('thickness_nm'); i = g.Pr_uC_cm2.idxmax()
        pk.append(dict(series=k, crystal=g.crystal.iloc[0], dopant=g.dopfam.iloc[0], n=len(g), t_at_maxPr=float(g.loc[i, 'thickness_nm']),
                       maxPr=float(g.Pr_uC_cm2.max()), Pr_thickest=float(g.Pr_uC_cm2.iloc[-1]), t_thickest=float(g.thickness_nm.iloc[-1]),
                       rho=round(stats.spearmanr(g.thickness_nm, g.Pr_uC_cm2).statistic, 2)))
R['Pr_t_series'] = pk

# ---------- 3. Within-study effects of anneal T and dopant content (Fisher-z pooled Spearman) ----------
def pooled(rows):
    z = np.array([np.arctanh(np.clip(r['rho'], -.999, .999)) for r in rows]); w = np.array([max(r['n'] - 3, 1) for r in rows])
    zbar = (w * z).sum() / w.sum(); se = 1 / np.sqrt(w.sum())
    return dict(k=len(rows), N=int(sum(r['n'] for r in rows)), rho=round(np.tanh(zbar), 3), ci=[round(np.tanh(zbar - 1.96 * se), 3), round(np.tanh(zbar + 1.96 * se), 3)])
d['series_T'] = d.study + '|' + d['stack'].fillna('') + '|' + d.thickness_nm.map(str) + '|' + d.dopant_conc.map(str) + '|' + d.dopant_ratio.fillna('')
ann = []
for k, g in d.dropna(subset=['anneal_C', 'Pr_uC_cm2']).groupby('series_T'):
    if g.anneal_C.nunique() >= 3 and g.study.nunique() == 1 and not g.study.iloc[0].startswith(('Yang', 'Chen')):
        r = stats.spearmanr(g.anneal_C, g.Pr_uC_cm2).statistic
        ann.append(dict(series=k, n=len(g), Tmin=g.anneal_C.min(), Tmax=g.anneal_C.max(), rho=round(r, 2), dPr=round(g.sort_values('anneal_C').Pr_uC_cm2.iloc[-1] - g.sort_values('anneal_C').Pr_uC_cm2.iloc[0], 1),
                        Ec_rho=None if g.Ec_MV_cm.notna().sum() < 3 else round(stats.spearmanr(g.anneal_C, g.Ec_MV_cm, nan_policy='omit').statistic, 2)))
R['anneal_series'] = ann; R['anneal_pooled'] = pooled(ann) if ann else None

d['series_c'] = d.study + '|' + d['stack'].fillna('') + '|' + d.thickness_nm.map(str) + '|' + d.anneal_C.map(str)
conc = []
for k, g in d.groupby('series_c'):
    g = g.dropna(subset=['Pr_uC_cm2'])
    cc = g.dopant_conc if g.dopant_conc.notna().sum() >= 3 else g.dopant_ratio.str.split(':').str[0].astype(float, errors='ignore') if 'dopant_ratio' in g and g.dopant_ratio.notna().sum() >= 3 else None
    if cc is None or pd.Series(cc).nunique() < 3 or g.study.iloc[0].startswith(('Yang', 'Chen')): continue
    cc = pd.to_numeric(cc, errors='coerce')
    if 'ALD cycle' in str(g.conc_unit.iloc[0]): cc = 1 / cc     # higher ratio = less Si
    i = g.Pr_uC_cm2.idxmax(); g2 = g.assign(c=cc).sort_values('c')
    Pmax = g.Pr_uC_cm2.max(); win = g2[g2.Pr_uC_cm2 >= 0.5 * Pmax]
    conc.append(dict(series=k, dopant=g[g.dopfam!='Other'].dopfam.mode().iloc[0] if (g.dopfam!='Other').any() else 'Other', crystal=g.crystal.iloc[0], n=len(g), c_opt=float(g.dopant_conc.loc[i]) if pd.notna(g.dopant_conc.loc[i]) else str(g.dopant_ratio.loc[i]),
                     unit=str(g.conc_unit.iloc[0]), Pmax=float(Pmax), Pmin=float(g.Pr_uC_cm2.min()), window_halfmax=[float(win.c.min()), float(win.c.max())] if 'ALD' not in str(g.conc_unit.iloc[0]) else None))
R['conc_series'] = conc

# ---------- 4. Multivariable model with study-clustered SEs ----------
def ols_cluster(df, y, X):
    Xm = np.column_stack([np.ones(len(df))] + [df[c].values.astype(float) for c in X]); Y = df[y].values
    b, *_ = np.linalg.lstsq(Xm, Y, rcond=None); res = Y - Xm @ b
    XtXi = np.linalg.pinv(Xm.T @ Xm); meat = np.zeros((Xm.shape[1],) * 2)
    for _, idx in df.groupby('study').indices.items():
        u = Xm[idx].T @ res[idx]; meat += np.outer(u, u)
    G, n, k = df.study.nunique(), len(df), Xm.shape[1]
    V = XtXi @ meat @ XtXi * G / (G - 1) * (n - 1) / (n - k); se = np.sqrt(np.diag(V))
    r2 = 1 - (res**2).sum() / ((Y - Y.mean())**2).sum()
    tval = b / se; pv = 2 * stats.t.sf(np.abs(tval), G - 1)
    return dict(coef=dict(zip(['const'] + X, np.round(b, 4))), se=dict(zip(['const'] + X, np.round(se, 4))), p=dict(zip(['const'] + X, pv)), r2=r2, n=n, G=G)
dm = d.copy()
for f in ['La', 'Si', 'Y', 'Al', 'Other']: dm[f'dop_{f}'] = (dm.dopfam == f).astype(int)
for f in ['PLD', 'CSD', 'Sputtering']: dm[f'dep_{f}'] = (dm.dep == f).astype(int)
groups = {'microstructure (epitaxial)': ['epi'], 'thickness (log t)': ['logt'], 'dopant': ['dop_La', 'dop_Si', 'dop_Y', 'dop_Al', 'dop_Other'],
          'deposition': ['dep_CSD', 'dep_Sputtering'], 'anneal T': ['anneal_C']}   # PLD collinear with epi -> folded into microstructure
R['models'] = {}
for y, lab in [('logEc', 'log10 Ec'), ('Pr_uC_cm2', 'Pr')]:
    allX = sum(groups.values(), [])
    dd = dm.dropna(subset=[y, 'logt', 'anneal_C'])
    full = ols_cluster(dd, y, allX)
    dR2 = {}
    for gname, cols in groups.items():
        red = ols_cluster(dd, y, [c for c in allX if c not in cols]); dR2[gname] = round(full['r2'] - red['r2'], 4)
    # bootstrap ΔR2 by resampling studies
    studies = dd.study.unique(); boot = {g: [] for g in groups}
    for _ in range(500):
        pick = rng.choice(studies, len(studies), replace=True)
        bd = pd.concat([dd[dd.study == s].assign(study=f'{s}_{i}') for i, s in enumerate(pick)])
        try:
            fr = ols_cluster(bd, y, allX)['r2']
            for gname, cols in groups.items(): boot[gname].append(fr - ols_cluster(bd, y, [c for c in allX if c not in cols])['r2'])
        except Exception: pass
    ci = {g: [round(np.percentile(v, 2.5), 3), round(np.percentile(v, 97.5), 3)] for g, v in boot.items()}
    M = np.array([boot[g] for g in groups]); top = {g: round(float((M.argmax(0) == i).mean()), 2) for i, g in enumerate(groups)}
    R['models'][lab] = dict(n=full['n'], studies=full['G'], r2=round(full['r2'], 3), coef=full['coef'], se=full['se'],
                            p={k: float(v) for k, v in full['p'].items()}, dR2=dR2, dR2_CI=ci, boot_top_share=top, n_boot=int(M.shape[1]))

# ---------- 5. Sensitivity ----------
sens = {}
def gap(df):
    a, b = df[df.epi == 1].Ec_MV_cm.median(), df[df.epi == 0].Ec_MV_cm.median(); return a / b
sens['Ec_ratio_all'] = round(gap(d), 2)
sens['Ec_ratio_no_digitized'] = round(gap(d[~d.figure_digitized]), 2) if d[~d.figure_digitized & (d.epi == 1)].Ec_MV_cm.notna().sum() > 3 else None
sens['Ec_ratio_only_HZO'] = round(gap(d[d.dopfam == 'Zr (HZO)']), 2)
sens['Ec_ratio_t_5_20nm'] = round(gap(d[(d.thickness_nm >= 5) & (d.thickness_nm <= 20)]), 2)
loo = [round(gap(d[d.study != s]), 2) for s in d.study.unique()]; sens['Ec_ratio_leave_one_study_out'] = [min(loo), max(loo)]
sens['n_epi_digitized_share'] = round(d[d.epi == 1].figure_digitized.mean(), 2)
# study-level medians (each study counts once)
sm = d.groupby(['study', 'crystal']).Ec_MV_cm.median().dropna().reset_index()
sens['study_level_Ec_median'] = {c: [round(g.Ec_MV_cm.median(), 2), int(len(g))] for c, g in sm.groupby('crystal')}
sens['MW_study_level'] = float(stats.mannwhitneyu(sm[sm.crystal == 'epitaxial'].Ec_MV_cm, sm[sm.crystal == 'polycrystalline'].Ec_MV_cm).pvalue)
# epitaxial-only: Ec-t slope within thickness series (mean of series slopes)
es = [s['slope'] for s in series_t if s['crystal'] == 'epitaxial']; ps = [s['slope'] for s in series_t if s['crystal'] == 'polycrystalline']
sens['series_slope_epi_mean'] = [round(np.mean(es), 2), round(np.std(es, ddof=1), 2), len(es)] if es else None
sens['series_slope_poly'] = ps
R['sensitivity'] = sens
# Pr-Ec relation
for c, g in d.dropna(subset=['Ec_MV_cm', 'Pr_uC_cm2']).groupby('crystal'):
    r = stats.spearmanr(g.Ec_MV_cm, g.Pr_uC_cm2); R[f'Pr_Ec_rho_{c}'] = [round(r.statistic, 2), float(r.pvalue), len(g)]

d.to_csv('hfo2_analysis_table.csv', index=False)
json.dump(R, open('results.json', 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=float)[:9000])
