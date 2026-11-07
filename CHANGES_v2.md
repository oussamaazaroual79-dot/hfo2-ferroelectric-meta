# Version 2 (November 2026)

New files:
- `analysis_v2.py` — electrode classification, linear mixed-effects models (random intercept per study, REML), thickness-window fixed-effects exponents, and per-series comparison of scaling models (constant, free power law, JKD n = -2/3, dead layer E_inf + A/t) by AICc.
- `figs_v2.py` — figures for the electrode effect and the thickness-window analysis.
- `results_v2.json`, `physical_model_comparison.csv`, `hfo2_model_table_v2.csv` — outputs.

Run: `python analysis.py && python analysis_v2.py && python figs.py && python figs_v2.py` (needs numpy, pandas, scipy, matplotlib).
