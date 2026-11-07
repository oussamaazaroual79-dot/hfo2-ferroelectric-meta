import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': 0.8, 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 300, 'figure.dpi': 120})
EPI, POLY = '#2a78d6', '#eb6834'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GRID = '#e6e5e1'
d = pd.read_csv('hfo2_analysis_table.csv'); R = json.load(open('results.json'))
full = pd.read_csv('hfo2_db_v4.csv')
def short(s): return s.split(',')[0].replace(' et al.', '') + ' ' + [x for x in s.split(',') if x.strip()[:4].isdigit() or x.strip()[-4:].isdigit()][0].strip()[-4:]
def grid(ax): ax.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)

# ---------- Fig 1: Ec vs thickness, pooled (a) and within-series (b) ----------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
def logaxes(ax):
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(3, 900); ax.set_ylim(0.6, 5.5)
    ax.set_xticks([3, 10, 30, 100, 300]); ax.set_xticklabels(['3', '10', '30', '100', '300'])
    ax.set_yticks([1, 2, 4]); ax.set_yticklabels(['1', '2', '4'])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Film thickness t (nm)'); grid(ax)
ax = axs[0]
for c, col, mk in [('polycrystalline', POLY, 'o'), ('epitaxial', EPI, 's')]:
    g = d[(d.crystal == c)].dropna(subset=['thickness_nm', 'Ec_MV_cm'])
    ax.scatter(g.thickness_nm, g.Ec_MV_cm, s=20, marker=mk, facecolor=col, edgecolor='white', lw=0.6, alpha=0.85, zorder=3, label=f'{c.capitalize()} (n = {len(g)})')
    f = R['Ec_t_pooled'][c]; x = np.logspace(np.log10(g.thickness_nm.min()), np.log10(min(g.thickness_nm.max(), 300)), 50)
    b = np.polyfit(np.log10(g.thickness_nm), np.log10(g.Ec_MV_cm), 1)
    ax.plot(x, 10**np.polyval(b, np.log10(x)), color=col, lw=1.6, zorder=4)
    ax.text(x[-1] * 1.1, 10**np.polyval(b, np.log10(x[-1])), f'{f["slope"]:+.2f}', color='#0b0b0b', va='center', fontsize=8)
logaxes(ax); ax.set_ylabel(r'Coercive field $E_c$ (MV cm$^{-1}$)')
ax.set_title('(a) All films, pooled fit', loc='left', fontsize=9); ax.legend(loc='lower left', fontsize=7, handletextpad=0.3)
ax = axs[1]
ser = d[~d.method.str.contains('review table', na=False)].dropna(subset=['thickness_nm', 'Ec_MV_cm'])
keys = [s['series'] for s in R['Ec_t_series']]
ep_cols = ['#2a78d6', '#184f95', '#6da7ec', '#104281', '#5598e7', '#256abf']; po_cols = ['#eb6834', '#c94f20', '#f29a73', '#a8401a']
ie = ip = 0
for s in R['Ec_t_series']:
    g = ser[ser.series_key == s['series']].sort_values('thickness_nm')
    if s['crystal'] == 'epitaxial': col, mk = ep_cols[ie % 6], 's'; ie += 1
    else: col, mk = po_cols[ip % 4], 'o'; ip += 1
    ax.plot(g.thickness_nm, g.Ec_MV_cm, color=col, lw=1.3, marker=mk, ms=3.5, mec='white', mew=0.4, zorder=3)
for sl, ls, lab, y0 in [(-2/3, (0, (4, 3)), r'$t^{-2/3}$', 4.6), (-1/2, (0, (1, 2)), r'$t^{-1/2}$', 5.2)]:
    x = np.array([4.5, 60]); ax.plot(x, y0 * (x / 4.5)**sl, color='#52514e', lw=1, ls=ls, zorder=2)
    ax.text(62, y0 * (60 / 4.5)**sl, lab, color='#52514e', fontsize=8, va='center')
ax.plot([], [], color=EPI, marker='s', ms=3.5, lw=1.3, label=f'Epitaxial series (k = {ie})'); ax.plot([], [], color=POLY, marker='o', ms=3.5, lw=1.3, label=f'Polycrystalline series (k = {ip})')
logaxes(ax); ax.set_title('(b) Within-series thickness scans', loc='left', fontsize=9); ax.legend(loc='lower left', fontsize=7, handletextpad=0.3)
fig.tight_layout(); fig.savefig('fig1_Ec_thickness.png'); fig.savefig('fig1_Ec_thickness.pdf'); plt.close()

# ---------- Fig 6: forest plot of series exponents ----------
rows = sorted(R['Ec_t_series'], key=lambda s: (s['crystal'] != 'epitaxial', s['slope']))
fig, ax = plt.subplots(figsize=(6.8, 3.8))
labs = []
for i, s in enumerate(rows):
    col = EPI if s['crystal'] == 'epitaxial' else POLY
    se = s['se'] if s['se'] == s['se'] else 0
    ax.errorbar(s['slope'], i, xerr=1.96 * se, fmt='s' if col == EPI else 'o', color=col, ms=5, mec='white', mew=0.5, elinewidth=1.2, capsize=0, zorder=3)
    st, stack = s['series'].split('|')[:2]
    sub = ' STO(110)' if '110' in stack else (' Si' if 'Si(001)' in stack else (' STO(001)' if 'STO' in stack else ''))
    dl = {'Zr': 'HZO', 'Other': 'Ge:HfO2'}.get(s['dopant'].split()[0], s['dopant'].split()[0] + ':HfO2')
    labs.append(f"{dl}{', PVD' if 'PVD' in stack else (', ALD' if 'TiN' in stack else '')}{',' + sub if sub else ''} – {short(st)}  ({s['tmin']:g}–{s['tmax']:g} nm)".replace('  (', ' (').replace(' –', ' –'))
n = len(rows)
for key, y, col in [('epitaxial t<=50nm', n + 0.6, EPI), ('polycrystalline t<=50nm', n + 1.6, POLY)]:
    f = R['FE_scaling'][key]; ax.errorbar(f['slope'], y, xerr=[[f['slope'] - f['ci'][0]], [f['ci'][1] - f['slope']]], fmt='D', color=col, ms=7, mec='#0b0b0b', mew=0.6, elinewidth=2, capsize=0, zorder=4)
labs += ['Pooled within-series, epitaxial (t ≤ 50 nm)', 'Pooled within-series, polycrystalline (t ≤ 50 nm)']
ax.set_yticks(list(range(n)) + [n + 0.6, n + 1.6]); ax.set_yticklabels(labs, fontsize=6.8)
for x, lab in [(-2/3, '−2/3 (JKD)'), (-0.5, '−1/2'), (0, '0')]:
    ax.axvline(x, color='#52514e', lw=0.9, ls=(0, (4, 3)) if x else '-', zorder=1); ax.text(x, -1.3, lab, ha='center', fontsize=7, color='#52514e')
ax.set_ylim(-1.8, n + 2.3); ax.invert_yaxis(); ax.set_xlim(-0.95, 0.12)
ax.set_xlabel(r'Scaling exponent $n$ in $E_c \propto t^{\,n}$ (±95% CI)'); ax.grid(True, axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig6_exponent_forest.png'); fig.savefig('fig6_exponent_forest.pdf'); plt.close()

# ---------- Fig 2: Pr vs thickness, within-series ----------
fig, ax = plt.subplots(figsize=(3.5, 3.0))
g = d[(d.crystal == 'polycrystalline')].dropna(subset=['thickness_nm', 'Pr_uC_cm2'])
ax.scatter(g.thickness_nm, g.Pr_uC_cm2, s=14, marker='o', facecolor='none', edgecolor=POLY, lw=0.7, alpha=0.8, zorder=2, label=f'Polycrystalline, all (n = {len(g)})')
mk = ['s', 'D', '^', 'v', 'P', 'X']
for i, s in enumerate([s for s in R['Pr_t_series'] if s['crystal'] == 'epitaxial']):
    st, stack = s['series'].split('|')[:2]
    gg = d[d.series_key == s['series']].dropna(subset=['thickness_nm', 'Pr_uC_cm2']).sort_values('thickness_nm')
    sub = 'STO(110)' if '110' in stack else ('Si' if 'Si(001)' in stack else 'STO(001)')
    ax.plot(gg.thickness_nm, gg.Pr_uC_cm2, color=CAT[i if i < 1 else i + 1] if False else [EPI, '#1baf7a', '#6250d6', '#eda100', '#e87ba4', '#008300'][i],
            lw=1.4, marker=mk[i], ms=4.5, mec='white', mew=0.5, zorder=3, label=f'{s["dopant"].split()[0]}, {short(st)}, {sub}')
ax.set_xscale('log'); ax.set_xticks([3, 10, 30, 100]); ax.set_xticklabels(['3', '10', '30', '100'])
ax.set_xlabel('Film thickness t (nm)'); ax.set_ylabel(r'Remanent polarization $P_r$ ($\mu$C cm$^{-2}$)'); grid(ax)
ax.set_ylim(0, 37); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(fontsize=6.3, loc='upper left', bbox_to_anchor=(1.01, 1.0), handlelength=1.6, labelspacing=0.5)
fig.set_size_inches(5.6, 3.0); fig.tight_layout(); fig.savefig('fig2_Pr_thickness.png'); fig.savefig('fig2_Pr_thickness.pdf'); plt.close()

# ---------- Fig 3: within-study composition windows and anneal ----------
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0))
def series(study_start, stack, **flt):
    g = d[d.study.str.startswith(study_start) & (d['stack'] == stack)]
    for k, v in flt.items(): g = g[g[k] == v]
    return g
a = axs[0, 0]   # La
g = series('Schroeder', 'TiN/La:HfO2/TiN', thickness_nm=10.0).sort_values('dopant_conc')
a.plot(g.dopant_conc, g.Pr_uC_cm2, color=POLY, marker='o', ms=4, lw=1.4, mec='white', mew=0.5, label='Polycryst., ALD (Schroeder 2018)')
for i, (stk, lab) in enumerate([('Pt/La:HfO2/LSMO/STO(001)', 'Epitaxial, STO (Song 2021)'), ('Pt/La:HfO2/LSMO/STO/Si(001)', 'Epitaxial, Si (Song 2021)')]):
    g = series('Song et al., ACS', stk).sort_values('dopant_conc')
    a.plot(g.dopant_conc, g.Pr_uC_cm2, color=[EPI, '#1baf7a'][i], marker=['s', 'D'][i], ms=4, lw=1.4, mec='white', mew=0.5, label=lab)
a.set_xlabel('La content (cat% / at%)'); a.set_title('(a) La', loc='left', fontsize=9)
a = axs[0, 1]   # Si
g = series('Richter', 'TiN/Si:HfO2/TiN', thickness_nm=36.0, anneal_C=800.0)
g = g[g.method == 'figure 7a'].copy(); g['r'] = g.dopant_ratio.str.split(':').str[0].astype(float); g['si'] = 100 / (g.r + 1); g = g.sort_values('si')
a.plot(g.si, g.Pr_uC_cm2, color=POLY, marker='o', ms=4, lw=1.4, mec='white', mew=0.5, label='Polycryst., ALD, 36 nm (Richter 2017)')
a.set_xlabel(r'SiO$_2$ ALD cycle fraction (%)'); a.set_title('(b) Si', loc='left', fontsize=9)
a = axs[1, 0]   # Zr
g = series('Liao', 'TaN/stacked HfO2-ZrO2/n+Si').sort_values('dopant_conc')
a.plot(g.dopant_conc, g.Pr_uC_cm2, color=POLY, marker='o', ms=4, lw=1.4, mec='white', mew=0.5, label='Polycryst., ALD, 10 nm (Liao 2023)')
a.set_xlabel('Zr fraction (%)'); a.set_title('(c) Zr', loc='left', fontsize=9)
a = axs[1, 1]   # anneal
for i, (st, stk, kw, lab, mk_) in enumerate([('Schroeder', 'TiN/La:HfO2/TiN', dict(thickness_nm=12.0), 'La:HfO$_2$, ALD (Schroeder 2018)', 'o'),
                                             ('Richter', 'TiN/Si:HfO2/TiN', dict(method='figure 2'), 'Si:HfO$_2$, ALD (Richter 2017)', '^'),
                                             ('Lyu et al., ACS', 'Pt/HZO/LSMO/STO(001)', dict(thickness_nm=9.2), 'HZO, epitaxial PLD (Lyu 2019)', 's')]):
    g = series(st, stk, **kw).dropna(subset=['anneal_C'])
    g = g[g.anneal_note.str.contains('series|RTA|N2', na=False) | (g.anneal_C == 800)] if st.startswith('Lyu') else g
    g = g[~g.anneal_note.str.contains('PO2 0.0|PO2 0.2|PO2 0.15', na=False)].groupby('anneal_C').Pr_uC_cm2.mean().reset_index().sort_values('anneal_C')
    a.plot(g.anneal_C, g.Pr_uC_cm2, color=[POLY, '#eda100', EPI][i], marker=mk_, ms=4, lw=1.4, mec='white', mew=0.5, label=lab)
a.set_xlabel(u'Anneal / growth temperature (°C)'); a.set_title('(d) Anneal temperature', loc='left', fontsize=9)
for a in axs.flat:
    a.set_ylabel(r'$P_r$ ($\mu$C cm$^{-2}$)'); grid(a); a.legend(fontsize=6.5, loc='upper left' if a is not axs[0, 0] else 'upper right'); a.set_ylim(bottom=0)
axs[0, 0].set_ylim(0, 36); axs[1, 1].set_ylim(0, 36); axs[0, 1].set_ylim(0, 18); axs[1, 0].set_ylim(0, 22)
fig.tight_layout(); fig.savefig('fig3_composition_anneal.png'); fig.savefig('fig3_composition_anneal.pdf'); plt.close()

# ---------- Fig 4: variance decomposition ----------
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
names = ['microstructure (epitaxial)', 'thickness (log t)', 'dopant', 'deposition', 'anneal T']
labels = ['Microstructure\n(epi vs poly)', 'Thickness', 'Dopant', 'Deposition\nmethod', 'Anneal T']
for a, (key, ttl) in zip(axs, [('log10 Ec', r'(a) log $E_c$'), ('Pr', r'(b) $P_r$')]):
    m = R['models'][key]; v = np.array([m['dR2'][n] for n in names])
    a.barh(range(5), v, color=[EPI if key.startswith('log') else POLY] * 5, height=0.62, zorder=3)
    for i, x in enumerate(v): a.text(x + 0.006, i, f'{x:.2f}', va='center', fontsize=8, color='#0b0b0b')
    a.set_yticks(range(5)); a.set_yticklabels(labels, fontsize=8); a.invert_yaxis(); a.set_xlim(0, 0.4)
    a.set_title(ttl, loc='left', fontsize=9); a.text(0.98, 0.97, f'model $R^2$ = {m["r2"]:.2f}\nn = {m["n"]} films, {m["studies"]} studies', transform=a.transAxes, ha='right', va='top', fontsize=7.5, color='#52514e')
    a.set_xlabel(r'Unique variance explained ($\Delta R^2$)'); a.grid(True, axis='x', color=GRID, lw=0.6, zorder=0); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig4_variance_decomposition.png'); fig.savefig('fig4_variance_decomposition.pdf'); plt.close()

# ---------- Fig 5: Pr vs Ec map ----------
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for c, col, mk_ in [('polycrystalline', POLY, 'o'), ('epitaxial', EPI, 's')]:
    g = d[d.crystal == c].dropna(subset=['Ec_MV_cm', 'Pr_uC_cm2'])
    ax.scatter(g.Ec_MV_cm, g.Pr_uC_cm2, s=22, marker=mk_, facecolor=col, edgecolor='white', lw=0.6, alpha=0.9, zorder=3, label=f'{c.capitalize()} (n = {len(g)})')
imp = full[full.physically_implausible.astype(bool)]
ax.scatter(imp.Ec_MV_cm, imp.Pr_uC_cm2, s=22, marker='x', color='#52514e', lw=1, zorder=3, label=f'Excluded, implausible (n = {len(imp)})')
ax.axhspan(55, 95, color='#f0efec', zorder=0); ax.text(4.85, 82, r'$P_r$ above theoretical $P_s$', ha='right', fontsize=7.5, color='#52514e')
ax.set_xlabel(r'$E_c$ (MV cm$^{-1}$)'); ax.set_ylabel(r'$P_r$ ($\mu$C cm$^{-2}$)'); ax.set_xlim(0, 5.0); ax.set_ylim(0, 92); grid(ax)
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
fig.set_size_inches(5.4, 3.0)
fig.tight_layout(); fig.savefig('fig5_Pr_Ec_map.png'); fig.savefig('fig5_Pr_Ec_map.pdf'); plt.close()
print('ok')
