import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('figs.py').read().split('# ---------- Fig 1')[0].replace("d = pd.read_csv('hfo2_analysis_table.csv'); R = json.load(open('results.json'))","").replace("full = pd.read_csv('hfo2_db_v4.csv')",""))
m = pd.read_csv('hfo2_model_table_v2.csv'); R = json.load(open('results_v2.json')); pm = pd.read_csv('physical_model_comparison.csv')
# ---------- Fig 7: electrode configuration (polycrystalline) ----------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]
p = m[m.epi == 0]
classes = [('Nitride/W symmetric', CAT[0], 'o'), ('Noble or mixed metal', CAT[2], '^'), ('Semiconductor bottom (MFIS)', CAT[1], 's'), ('Not reported', '#9a9893', 'D')]
for c, col, mk in classes:
    g = p[p.electrode == c]
    ax.scatter(g.thickness_nm, g.Ec_MV_cm, s=20, marker=mk, facecolor=col, edgecolor='white', lw=.6, alpha=.9, zorder=3, label=f'{c} (n = {len(g)})')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(3, 900); ax.set_ylim(.6, 5.5)
ax.set_xticks([3, 10, 30, 100, 300]); ax.set_xticklabels(['3', '10', '30', '100', '300']); ax.set_yticks([1, 2, 4]); ax.set_yticklabels(['1', '2', '4'])
for a in (ax.xaxis, ax.yaxis): a.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel('Film thickness t (nm)'); ax.set_ylabel(r'Coercive field $E_c$ (MV cm$^{-1}$)'); grid(ax)
ax.legend(fontsize=6.5, loc='upper right'); ax.text(-.16, 1.02, '(a)', transform=ax.transAxes, fontweight='bold')
ax = axs[1]
co = R['LMM_poly_electrode']['coef']
lab = [('el_MFIS', 'Semiconductor\nbottom (MFIS)'), ('el_noble', 'Noble or\nmixed metal'), ('el_NR', 'Not reported')]
for i, (k, l) in enumerate(lab):
    b, (lo, hi) = co[k]['b'], co[k]['ci']
    f = lambda v: (10**v - 1) * 100
    ax.errorbar(f(b), i, xerr=[[f(b) - f(lo)], [f(hi) - f(b)]], fmt='o', color=CAT[1] if k == 'el_MFIS' else '#52514e', capsize=3, ms=5)
ax.axvline(0, color='#52514e', lw=.8, ls='--'); ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in lab], fontsize=8)
ax.set_ylim(-.6, 2.6); ax.invert_yaxis(); ax.set_xlabel('Change in $E_c$ vs TiN/TaN/W (%)'); grid(ax)
ax.text(-.42, 1.02, '(b)', transform=ax.transAxes, fontweight='bold')
fig.tight_layout(); fig.savefig('paper/Fig7.pdf'); fig.savefig('paper/Fig7.png'); plt.close()

# ---------- Fig 8: thickness window and physical models ----------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
ax = axs[0]
for c, col, mk in [('epitaxial', EPI, 's'), ('polycrystalline', POLY, 'o')]:
    g = pm[pm.crystal == c]
    ax.scatter(g.tmax, -g.slope, s=34, marker=mk, facecolor=col, edgecolor='white', lw=.6, zorder=3, label=c.capitalize())
ax.axhline(2 / 3, color='#52514e', ls='--', lw=.9); ax.text(11, .685, 'JKD exponent 2/3', fontsize=7, color='#52514e', ha='left')
ax.set_xscale('log'); ax.set_xlabel(r'Largest thickness in the series $t_{max}$ (nm)'); ax.set_ylabel('Apparent exponent  $-n$')
ax.set_xticks([10, 30, 100, 300]); ax.set_xticklabels(['10', '30', '100', '300']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(0, .85); grid(ax); ax.legend(fontsize=7, loc='lower left')
rho, pv = R['slope_vs_tmax_spearman']; ax.text(.97, .99, f'Spearman ρ = {-rho:.2f}, p = {pv:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=7)
ax.text(-.16, 1.02, '(a)', transform=ax.transAxes, fontweight='bold')
ax = axs[1]
W = R['FE_window']; We = R['FE_window_epi']
items = [('All series, t ≤ 20 nm', W['t<=20nm']), ('Epitaxial, t ≤ 20 nm', We['t<=20nm']), ('Polycrystalline, t ≤ 20 nm', We['t<=20nm_poly']), ('All series, t > 20 nm', W['t>20nm'])]
for i, (l, v) in enumerate(items):
    ax.errorbar(-v['b'], i, xerr=[[-v['b'] + v['ci'][1]], [-v['ci'][0] + v['b']]], fmt='o', color=(EPI if l.startswith('Epi') else POLY if l.startswith('Poly') else '#0b0b0b' if '≤' in l else '#9a9893'), capsize=3, ms=5)
    ax.text(.02, i - .28, f"{v['n_series']} series, {v['n_films']} films", fontsize=6.5, color='#52514e')
ax.axvline(2 / 3, color='#52514e', ls='--', lw=.9); ax.set_yticks(range(4)); ax.set_yticklabels([l for l, _ in items], fontsize=7.5)
ax.set_ylim(-.6, 3.6); ax.invert_yaxis(); ax.set_xlim(0, .8); ax.set_xlabel('Within-series exponent  $-n$'); grid(ax)
ax.text(-.62, 1.02, '(b)', transform=ax.transAxes, fontweight='bold')
fig.tight_layout(); fig.savefig('paper/Fig8.pdf'); fig.savefig('paper/Fig8.png'); plt.close()
