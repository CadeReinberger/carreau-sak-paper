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
| 2  | Blasius / Sakiadis schematics | `schematics_pdf_cropped.pdf` | **PowerPoint** — source `.pptx` not in repo; only the exported `schematics_pdf.pdf` survives here |
| 3  | Shooting separatrix, α = 0.3 | `fig3.eps` | `fig_3.m` |
| 4  | Thinning asymptotic, α = 0.1 | `new_thinning_asymptotic.eps` | `figs_4_7.py` |
| 5  | κ_p vs α (Blasius + Sakiadis) | `fig5.eps` | `fig_5.py` |
| 6  | Shooting separatrix, α = 1.4 | `fig6.eps` | `fig_6.m` |
| 7  | 4th-derivative discontinuity, α = 1.4 | `new_thickening_deriv.eps` | `figs_4_7.py` |
| 8  | Wall shear — Sakiadis (top) + Blasius (bottom) | `new_shear.eps` + `new_shear_blas.eps` | `figs_8_11_sakiadis.py` + `figs_8_11_blasius.py` |
| 9  | Wall-shear front error, α = 0.3 | `rel_err.eps` | `fig_9.py` |
| 10 | Correction function c(α) | `fig7.eps` | `fig_10.py` |
| 11 | Velocity profiles — Blasius (top) + Sakiadis (bottom) | `new_profiles_blas.eps` + `new_profiles.eps` | `figs_8_11_blasius.py` + `figs_8_11_sakiadis.py` |

## Notes / caveats

- **`fig_10.py` writes `fig7.eps`.** The output keeps its legacy name; in `transfer/main.tex`
  that image is **Figure 10**. Left unchanged so the paper still compiles.
- **`figs_8_11_sakiadis.py`** (was `cplots.py`) → `new_shear.eps` + `new_profiles.eps`.
  **`figs_8_11_blasius.py`** (was `bplots.py`) → `new_shear_blas.eps` + `new_profiles_blas.eps`.
  Each provides one row-block of both Fig 8 and Fig 11.
- **`figs_4_7.py`** (was `new_thickening_deriv.py`, imported from the `watanabe` repo)
  produces both `new_thinning_asymptotic.eps` (Fig 4) and `new_thickening_deriv.eps` (Fig 7).
- Regenerated outputs live here; the paper reads its own copies in `transfer/`, so after
  regenerating a figure, copy the updated file into `transfer/`.
- `archived/` holds superseded scripts/images (older PNG-variant and pre-`fig7.py`
  c(α) plots) kept for reference.

## Known fidelity gaps (scripts vs. the committed `transfer/` images)

- `fig_10.py` currently saves `fig7.png` with the Blasius curve commented out, whereas
  the paper's `fig7.eps` shows both Sakiadis and Blasius. Re-enable the Blasius line and
  the `.eps` save to reproduce it exactly.
