# HfO2 ferroelectric films: literature meta-analysis

Dataset and code for the manuscript *"Thickness scaling of the coercive field in ferroelectric HfO2-based films is not universal: a meta-analysis of 237 films that separates within-study trends from between-study scatter"* (O. Azaroual, submitted).

## Contents
- `data/hfo2_ferroelectric_films.csv` – 237 films from 24 sources. One row per film: stack, dopant and concentration, thickness, deposition, thermal budget, microstructure, Pr, Ec, source, original reference, extraction method (table / text / digitized figure) and quality flags (`physically_implausible`, `unsaturated`, `overlap_flag`).
- `data/hfo2_analysis_table.csv` – the 219 films used in the analysis, with derived variables.
- `data/results.json` – all numerical results reported in the paper.
- `code/analysis.py` – statistics (pooled and within-series power-law fits, fixed-effects exponents with series bootstrap, multivariable models with study-clustered errors, sensitivity analyses).
- `code/figs.py` – figures.

## Reproduce
```
cd code
python analysis.py
python figs.py
```
Requires Python 3 with pandas, numpy, scipy and matplotlib.

## Conventions
2Pr and 2Ec values were halved; Ec = Vc/t where only a coercive voltage was given; woken-up values are stored as Pr when both states were reported. Values digitized from figures carry an uncertainty of about ±5 %. Please cite the original articles listed in the `source` column when reusing individual values.

## License
Data: CC BY 4.0. Code: MIT.
