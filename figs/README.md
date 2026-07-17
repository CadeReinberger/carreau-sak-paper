# Figure code — Carreau/Sakiadis paper

Code for the figures in the **`transfer/` paper** (`transfer/main.tex`, the current
JFM-format manuscript). This is the canonical version; the repo-root `main.tex` is an
older draft and its figures are out of date.

One script per figure. Scripts that produce two related figures are named
`figs_<x>_<y>...`. Figure numbers below are the numbers in `transfer/main.tex`.

Shared numerics live in `ez_numerics.py` (Sakiadis) and `ez_numerics_blas.py` (Blasius);
the plotting scripts import these.

| Fig | Description | Output file(s) | Script |
|-----|-------------|----------------|--------|
| 1  | Rheology curves (thinning / thickening) | `rheo_thin/thick_cropped_rotated.pdf` | **PowerPoint** — `rheo.pptx` (no code) |
| 2  | Blasius / Sakiadis schematics | `schematics_pdf_cropped.pdf` | **PowerPoint** — source `.pptx` I can't find. Ask Steve, I think he made them. Only the `schematics_pdf.pdf` survives here |
| 3  | Shooting separatrix, α = 0.3 | `fig3.eps` | `fig_3.m` |
| 4  | Thinning asymptotic, α = 0.1 | `new_thinning_asymptotic.eps` | `figs_4_7.py` |
| 5  | κ_p vs α (Blasius + Sakiadis) | `fig5.eps` | `fig_5.py` |
| 6  | Shooting separatrix, α = 1.4 | `fig6.eps` | `fig_6.m` |
| 7  | 4th-derivative discontinuity, α = 1.4 | `new_thickening_deriv.eps` | `figs_4_7.py` |
| 8  | Wall shear — Sakiadis (top) + Blasius (bottom) | `new_shear.eps` + `new_shear_blas.eps` | `figs_8_11_sakiadis.py` + `figs_8_11_blasius.py` |
| 9  | Wall-shear front error, α = 0.3 | `rel_err.eps` | `fig_9.py` |
| 10 | Correction function c(α) | `fig10.eps` | `fig_10.py` |
| 11 | Velocity profiles — Blasius (top) + Sakiadis (bottom) | `new_profiles_blas.eps` + `new_profiles.eps` | `figs_8_11_blasius.py` + `figs_8_11_sakiadis.py` |
