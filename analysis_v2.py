"""Level-1 re-analysis: electrodes, process, mixed-effects model, physical-model comparison."""
import json, re, numpy as np, pandas as pd
from scipy import stats, optimize

rng = np.random.default_rng(42)
db = pd.read_csv('hfo2_db_v4.csv')
db['physically_implausible'] = db.physically_implausible.astype(bool)
db['unsaturated'] = db.unsaturated.fillna(False).astype(bool)
d = db[~db.physically_implausible & ~db.unsaturated].copy()
dep_map = {'ALD': 'ALD', 'TALD': 'ALD', 'PEALD': 'ALD', 'PAALD': 'ALD', 'PLD': 'PLD', 'CSD': 'CSD', 'Sputtering': 'Sputtering'}
d['dep'] = d.deposition.map(dep_map)
fam = lambda x: 'Zr (HZO)' if x in ('Zr', 'Zr+La interlayers') else (x if x in ('La', 'Si', 'Y', 'Al') else 'Other')
d['dopfam'] = d.dopant.map(fam)
d['epi'] = (d.crystal == 'epitaxial').astype(int)
d['study'] = d.source.str.split('(').str[0].str.strip()
d['logt'] = np.log10(d.thickness_nm); d['logEc'] = np.log10(d.Ec_MV_cm)
d['series_key'] = d.study + '|' + d['stack'].fillna('') + '|' + d.method.fillna('') + '|' + d.anneal_C.map(str) + '|' + d.dopant_conc.map(str) + '|' + d.architecture.fillna('')

# ---------- 1. Electrode classes ----------
def electrode(s):
    """Classify by the layers directly above and below the ferroelectric layer."""
    s = str(s)
    if '/' not in s: return 'Not reported'
    parts = [p.strip() for p in s.split('/')]
    idx = next((i for i, p in enumerate(parts) if re.search(r'HfO2|HZO|HfZrO|\[HfO2', p)), None)
    if idx is None or idx == 0: return 'Not reported'          # no top electrode stated
    if parts[idx].startswith('[HfO2') or parts[idx].startswith('(HfO2'):
        pass
    top = parts[idx - 1]
    rest = parts[idx + 1:]
    while rest and re.search(r'ZrO2|HfO2', rest[0]): rest = rest[1:]   # skip further ferroelectric sublayers
    if not rest: return 'Not reported'
    bottom = rest[0]
    if re.match(r'^(n\+|p\+)?(Si|Ge)(\(|$)|^SiOx$|^SiO2$|^IL$', bottom): return 'Semiconductor bottom (MFIS)'
    if re.match(r'^(LSMO|LNO|ITO)', bottom): return 'Oxide bottom (LSMO/LNO/ITO)'
    nitr = {'TiN', 'TaN', 'W'}
    if top in nitr and bottom in nitr: return 'Nitride/W symmetric'
    return 'Noble or mixed metal'
d['electrode'] = d['stack'].map(electrode)
R = {}
R['electrode_counts'] = {k: {'n': int(len(g)), 'studies': int(g.study.nunique()), 'epi': int(g.epi.sum()),
                             'Ec_med': round(float(g.Ec_MV_cm.median()), 2) if g.Ec_MV_cm.notna().any() else None,
                             't_med': round(float(g.thickness_nm.median()), 1)}
                         for k, g in d.groupby('electrode')}
R['electrode_x_crystal'] = pd.crosstab(d.electrode, d.crystal).to_dict()

# ---------- 2. Random-intercept linear mixed model (REML), hand-written ----------
def lmm(df, y, X, group):
    """y = Xb + u_g + e ; u ~ N(0, s2u), e ~ N(0, s2e). REML via profiling of lambda = s2u/s2e."""
    df = df.dropna(subset=[y] + X)
    keep = []
    for c in X:
        M = np.column_stack([np.ones(len(df))] + [df[k].values.astype(float) for k in keep + [c]])
        if np.linalg.matrix_rank(M) == M.shape[1]: keep.append(c)
    dropped = [c for c in X if c not in keep]; X = keep
    Y = df[y].values; Xm = np.column_stack([np.ones(len(df))] + [df[c].values.astype(float) for c in X])
    G = pd.factorize(df[group])[0]; ng = G.max() + 1; n, p = Xm.shape
    Z = np.zeros((n, ng)); Z[np.arange(n), G] = 1
    def parts(lam):
        V = np.eye(n) + lam * Z @ Z.T
        Vi = np.linalg.inv(V)
        XtVi = Xm.T @ Vi; A = XtVi @ Xm
        b = np.linalg.solve(A, XtVi @ Y); r = Y - Xm @ b
        s2 = float(r @ Vi @ r) / (n - p)
        ll = -0.5 * ((n - p) * np.log(s2) + np.linalg.slogdet(V)[1] + np.linalg.slogdet(A)[1] + (n - p))
        return ll, b, s2, A
    res = optimize.minimize_scalar(lambda l: -parts(np.exp(l))[0], bounds=(-12, 6), method='bounded')
    lam = float(np.exp(res.x)); ll, b, s2e, A = parts(lam)
    cov = s2e * np.linalg.inv(A); se = np.sqrt(np.diag(cov))
    dof = max(ng - 1, 1)  # same convention as the cluster-robust OLS (G-1)
    names = ['Intercept'] + X
    coef = {nm: dict(b=round(float(bb), 4), se=round(float(ss), 4), p=float(2 * stats.t.sf(abs(bb / ss), dof)),
                     ci=[round(float(bb - stats.t.ppf(.975, dof) * ss), 3), round(float(bb + stats.t.ppf(.975, dof) * ss), 3)])
            for nm, bb, ss in zip(names, b, se)}
    s2u = lam * s2e
    # marginal / conditional R2 (Nakagawa & Schielzeth)
    vf = float(np.var(Xm @ b))
    return dict(coef=coef, n=int(n), groups=int(ng), sigma2_study=round(s2u, 4), sigma2_resid=round(s2e, 4),
                ICC=round(s2u / (s2u + s2e), 3), R2_marginal=round(vf / (vf + s2u + s2e), 3),
                R2_conditional=round((vf + s2u) / (vf + s2u + s2e), 3), reml_ll=round(float(ll), 2), dropped_collinear=dropped)

m = d.dropna(subset=['logEc', 'logt']).copy()
m['logt_c'] = m.logt - m.logt.mean()
m['logt_x_epi'] = m.logt_c * m.epi
for k in ['La', 'Si', 'Y', 'Al', 'Other']: m[f'dop_{k}'] = (m.dopfam == k).astype(int)
for k in ['CSD', 'Sputtering']: m[f'dep_{k}'] = (m.dep == k).astype(int)
m['ann_100C'] = (m.anneal_C - m.anneal_C.median()) / 100
m['el_MFIS'] = (m.electrode == 'Semiconductor bottom (MFIS)').astype(int)
m['el_noble'] = (m.electrode == 'Noble or mixed metal').astype(int)
m['el_NR'] = (m.electrode == 'Not reported').astype(int)
R['logt_mean'] = round(float(m.logt.mean()), 3)

R['LMM_M0_thickness_crystal'] = lmm(m, 'logEc', ['logt_c', 'epi', 'logt_x_epi'], 'study')
m['logt_poly'] = m.logt_c * (1 - m.epi); m['logt_epi'] = m.logt_c * m.epi
R['LMM_M0b_separate_slopes'] = lmm(m, 'logEc', ['epi', 'logt_poly', 'logt_epi'], 'study')
R['LMM_M0b_series'] = lmm(m, 'logEc', ['epi', 'logt_poly', 'logt_epi'], 'series_key')
R['LMM_M1_full'] = lmm(m.dropna(subset=['ann_100C']), 'logEc',
                       ['logt_c', 'epi', 'logt_x_epi', 'dop_La', 'dop_Si', 'dop_Y', 'dop_Al', 'dop_Other',
                        'dep_CSD', 'dep_Sputtering', 'ann_100C', 'el_MFIS', 'el_noble', 'el_NR'], 'study')
# polycrystalline only: electrodes are not confounded with epitaxy there
pm = m[m.epi == 0]
R['LMM_poly_electrode'] = lmm(pm, 'logEc', ['logt_c', 'el_MFIS', 'el_noble', 'el_NR'], 'study')
R['LMM_poly_full'] = lmm(pm.dropna(subset=['ann_100C']), 'logEc',
                         ['logt_c', 'dop_La', 'dop_Si', 'dop_Y', 'dop_Al', 'dop_Other', 'dep_CSD', 'dep_Sputtering', 'ann_100C',
                          'el_MFIS', 'el_noble', 'el_NR'], 'study')
# study-as-series random intercept (finer grouping)
R['LMM_M0_series'] = lmm(m, 'logEc', ['logt_c', 'epi', 'logt_x_epi'], 'series_key')

# Within-study electrode contrasts (same study, different electrode class)
wc = []
for s, g in m.groupby('study'):
    if g.electrode.nunique() >= 2:
        wc.append(dict(study=s, classes={k: [round(float(x.Ec_MV_cm.median()), 2), int(len(x)), round(float(x.thickness_nm.median()), 1)] for k, x in g.groupby('electrode')}))
R['within_study_electrode'] = wc

# ---------- 3. Physical-model comparison per thickness series ----------
ser = m[~m.method.str.contains('review table', na=False)]
ser = ser[ser.groupby('series_key').thickness_nm.transform('nunique') >= 4]
def aicc(rss, n, k):
    a = n * np.log(rss / n) + 2 * k
    return a + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf
models = {}
rows = []
for k, g in ser.groupby('series_key'):
    t = g.thickness_nm.values.astype(float); E = g.Ec_MV_cm.values.astype(float); n = len(t)
    if n < 4: continue
    out = {}
    # (a) constant
    out['constant'] = (np.sum((np.log(E) - np.log(E).mean()) ** 2), 1)
    # (b) free power law (log-log OLS)
    s, i, *_ = stats.linregress(np.log(t), np.log(E)); out['power_free'] = (np.sum((np.log(E) - (i + s * np.log(t))) ** 2), 2)
    # (c) JKD-fixed exponent -2/3
    i2 = np.mean(np.log(E) + 2 / 3 * np.log(t)); out['JKD_-2/3'] = (np.sum((np.log(E) - (i2 - 2 / 3 * np.log(t))) ** 2), 1)
    # (d) dead layer: Ec = Ecinf + A/t (fit in log space, Ecinf>0)
    def f_dl(p): return np.log(np.clip(p[0] + p[1] / t, 1e-6, None))
    try:
        A0 = np.polyfit(1 / t, E, 1)
        pr = optimize.least_squares(lambda p: f_dl(p) - np.log(E), x0=[max(A0[1], .1), A0[0]], bounds=([1e-3, -50], [20, 200]))
        out['dead_layer'] = (np.sum(pr.fun ** 2), 2); dl = pr.x
    except Exception:
        dl = [np.nan, np.nan]
    a = {kk: aicc(max(v[0], 1e-12), n, v[1] + 1) for kk, v in out.items()}
    amin = min(a.values()); w = {kk: np.exp(-(v - amin) / 2) for kk, v in a.items()}; W = sum(w.values())
    rows.append(dict(series=k.split('|')[0] + ' | ' + k.split('|')[1], crystal=g.crystal.iloc[0], electrode=g.electrode.iloc[0], n=n,
                     tmin=float(t.min()), tmax=float(t.max()), slope=round(s, 3),
                     Ec_inf=round(float(dl[0]), 2), A_MVnm=round(float(dl[1]), 2),
                     best=min(a, key=a.get), **{f'w_{kk}': round(v / W, 3) for kk, v in w.items()}))
pm_df = pd.DataFrame(rows)
pm_df.to_csv('physical_model_comparison.csv', index=False)
R['physical_models'] = dict(n_series=int(len(pm_df)), best_counts=pm_df.best.value_counts().to_dict(),
                            mean_weights={c: round(float(pm_df[c].mean()), 3) for c in pm_df.columns if c.startswith('w_')},
                            JKD_slope_in_CI=None)
# Are within-series exponents consistent with -2/3?  (from v1 results.json)
try:
    v1 = json.load(open('results.json'))
    R['v1_fe'] = {k: v1[k] for k in v1 if k.startswith('Ec_t_FE')}
except Exception:
    pass
# global fixed-effects slope, test against -2/3 and -1 (dead-layer limit for t >> d)
fe = ser.copy()
x = fe.logt - fe.groupby('series_key').logt.transform('mean'); y = fe.logEc - fe.groupby('series_key').logEc.transform('mean')
b = float((x * y).sum() / (x * x).sum())
keys = fe.series_key.unique()
boots = []
for _ in range(2000):
    ks = rng.choice(keys, len(keys)); xx = []; yy = []
    for kk in ks:
        gg = fe[fe.series_key == kk]; xx.append(gg.logt - gg.logt.mean()); yy.append(gg.logEc - gg.logEc.mean())
    xx = pd.concat(xx); yy = pd.concat(yy); boots.append(float((xx * yy).sum() / (xx * xx).sum()))
R['FE_slope_4pt_series'] = dict(b=round(b, 3), ci=[round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                                n_series=int(len(keys)), n_films=int(len(fe)), P_boot_le_minus_2_3=round(float(np.mean(np.array(boots) <= -2 / 3)), 4))
# thickness-window dependence of the apparent exponent
R['slope_vs_tmax_spearman'] = [round(float(x), 3) for x in stats.spearmanr(pm_df.tmax, pm_df.slope)]
allser = m[~m.method.str.contains('review table', na=False)]
def fe_win(sub, minpts=3):
    sub = sub[sub.groupby('series_key').thickness_nm.transform('nunique') >= minpts]
    keys = sub.series_key.unique()
    def sl(g):
        x = g.logt - g.groupby('series_key').logt.transform('mean'); y = g.logEc - g.groupby('series_key').logEc.transform('mean')
        return float((x * y).sum() / (x * x).sum())
    b = sl(sub); bs = []
    for _ in range(2000):
        ks = rng.choice(keys, len(keys))
        bs.append(sl(pd.concat([sub[sub.series_key == k].assign(series_key=f'{k}#{i}') for i, k in enumerate(ks)])))
    return dict(b=round(b, 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], n_series=int(len(keys)), n_films=int(len(sub)))
R['FE_window'] = {'t<=20nm': fe_win(allser[allser.thickness_nm <= 20]), 't>20nm': fe_win(allser[allser.thickness_nm > 20])}
R['FE_window_epi'] = {'t<=20nm': fe_win(allser[(allser.thickness_nm <= 20) & (allser.epi == 1)]), 't<=20nm_poly': fe_win(allser[(allser.thickness_nm <= 20) & (allser.epi == 0)])}
json.dump(R, open('results_v2.json', 'w'), indent=1, default=float)
m.to_csv('hfo2_model_table_v2.csv', index=False)
print(json.dumps(R, indent=1, default=float)[:9000])
print(pm_df.to_string())
