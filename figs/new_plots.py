import numpy as np
import ez_numerics
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.cm as mpl_colormap
from matplotlib.lines import Line2D

# ════════════════════════════════════════════════════════════════════════════
#  PARAMETERS  — easy to change
# ════════════════════════════════════════════════════════════════════════════
ALPHAS = [0.3, 0.8, 1.4]

# De values used for shear curves — one loglog line each per panel.
# Adjust per-alpha if you want different ranges for each.
DES = {
    0.3: np.logspace(1, 5, num=5),
    0.8: np.logspace(1, 5, num=5),
    1.4: np.logspace(1, 5, num=5),
}

# Re_x range and resolution for shear curves
RES = np.logspace(0, 8, num=51)

# delta fractions relative to delta_star used for the 4 profile columns
DELTA_FRACS  = [0.01, 0.25, 1.0, 10.0]
DELTA_LABELS = [r'$\delta = 0.01\,\delta_*$',
                r'$\delta = 0.25\,\delta_*$',
                r'$\delta = \delta_*$',
                r'$\delta = 10\,\delta_*$']
# One color per delta fraction — consistent across both figures
DELTA_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

KAPPA_N = 0.4437483133688610511198328438501
DPI     = 200
FS      = 15      # base font size
LW      = 2.5     # base line width
# ════════════════════════════════════════════════════════════════════════════


# ── Step 1: power-law kappas and delta_star per alpha ────────────────────────
print('Step 1: kappa_p and delta_star ...')
kappa_p_cache    = {}
delta_star_cache = {}
for alpha in ALPHAS:
    kp = ez_numerics.compute_pl_kappa_default(alpha)
    kappa_p_cache[alpha] = kp
    # Intersection of Newtonian and power-law shear stress: tau_n = tau_pl
    #   => delta_star = (kappa_p^alpha / kappa_n)^(2*(1+alpha)/(1-alpha))
    ds = (kp**alpha / KAPPA_N) ** (2*(1+alpha)/(1-alpha))
    delta_star_cache[alpha] = ds
    print(f'  alpha={alpha}  kappa_p={kp:.5f}  delta_star={ds:.4g}')


# ── Step 2: Carreau kappas for each (alpha, delta) in the 3×4 profile grid ──
print('\nStep 2: Carreau kappas for the 12 profile cells ...')
profile_kc = {}   # (alpha, frac) -> kappa_c
for alpha in ALPHAS:
    for frac in DELTA_FRACS:
        delta = frac * delta_star_cache[alpha]
        pd = ez_numerics.problem_data(alpha, delta, 0)
        kc = ez_numerics.compute_carreau_kappa_default(pd)
        profile_kc[(alpha, frac)] = kc
        print(f'  alpha={alpha}  frac={frac:5.2f}  delta={delta:.4g}  kc={kc:.5f}')


# ── Step 3: shear curve data (the slow loop) ─────────────────────────────────
print('\nStep 3: Carreau shear curves (slow — one BVP per Re point) ...')
shear_tau = {}   # (alpha, De) -> tau array over RES
for alpha in ALPHAS:
    print(f'  alpha={alpha}')
    for De in tqdm(DES[alpha], desc=f'    De loop alpha={alpha}'):
        arr = np.zeros(len(RES))
        for (i, Re) in enumerate(RES):
            delta = De**2 / Re
            pd  = ez_numerics.problem_data(alpha, delta, 0)
            kc  = ez_numerics.compute_carreau_kappa_default(pd)
            arr[i] = kc * Re**-.5 * (1 + kc**2 * De**2 / Re)**(.5*(alpha - 1))
        shear_tau[(alpha, De)] = arr

tau_n_shear = KAPPA_N * RES**-.5   # Newtonian shear (same for all alpha)


# ── Step 4: Figure 1 — shear plot (1×3) ──────────────────────────────────────
print('\nStep 4: Building new_shear.png ...')

de_cmap = mpl_colormap.get_cmap('plasma')

fig_s, axes_s = plt.subplots(1, 3, figsize=(20, 5.5))

for (col_idx, alpha) in enumerate(ALPHAS):
    ax  = axes_s[col_idx]
    des = DES[alpha]
    ds  = delta_star_cache[alpha]

    # De colors: evenly spaced through plasma, avoiding the very bright/dark ends
    de_colors = [de_cmap(0.15 + 0.70 * i / max(len(des) - 1, 1))
                 for i in range(len(des))]

    # Newtonian
    ax.loglog(RES, tau_n_shear, 'k-', linewidth=LW, label='Newtonian')

    # Carreau curves
    for (j, De) in enumerate(des):
        ax.loglog(RES, shear_tau[(alpha, De)], '-',
                  color=de_colors[j], linewidth=LW,
                  label=f'$De = {De:.0f}$')

    # Points corresponding to each velocity-profile delta value
    # For fixed delta: Re_x = De^2/delta, tau = kc * Re_x^(-1/2) * (1+kc^2*delta)^((alpha-1)/2)
    for (frac, dlabel, dcol) in zip(DELTA_FRACS, DELTA_LABELS, DELTA_COLORS):
        delta  = frac * ds
        kc     = profile_kc[(alpha, frac)]
        Re_pts, tau_pts = [], []
        for De in des:
            Re_x   = De**2 / delta
            tau_pt = kc * Re_x**-.5 * (1 + kc**2 * delta)**(.5*(alpha - 1))
            Re_pts.append(Re_x)
            tau_pts.append(tau_pt)
        ax.loglog(Re_pts, tau_pts, 'o',
                  color=dcol, markersize=10,
                  markeredgecolor='k', markeredgewidth=0.7,
                  label=dlabel, zorder=6)

    ax.set_xlabel(r'$Re_x$', fontsize=FS + 1)
    ax.set_ylabel(r'$\bar{\tau}_w$', fontsize=FS + 1)
    ax.tick_params(labelsize=FS - 1)

    # alpha annotation inside panel (no figure/axes title)
    ax.text(0.97, 0.97, f'$\\alpha = {alpha}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=FS + 2,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))

    ax.legend(fontsize=FS - 3, loc='lower left')

fig_s.tight_layout()
fig_s.savefig('new_shear.png', dpi=DPI, bbox_inches='tight')
plt.close(fig_s)
print('  Saved new_shear.png')


# ── Step 5: velocity profile BVP solutions ───────────────────────────────────
print('\nStep 5: Computing velocity profiles ...')

# Newtonian profile (alpha=1 power-law BVP — same for all panels)
eta_n_p, y_n_p = ez_numerics.compute_power_law_bvp_soln(
    1, ez_numerics.make_pl_default_unstable_solution_data(1))
ux_n_p = y_n_p[1, :]

# Power-law profile per alpha (raw BVP eta, rescaled per panel)
pl_eta_raw = {}
pl_ux_raw  = {}
for alpha in ALPHAS:
    eta_p, yp = ez_numerics.compute_power_law_bvp_soln(
        alpha, ez_numerics.make_pl_default_unstable_solution_data(alpha))
    pl_eta_raw[alpha] = eta_p
    pl_ux_raw[alpha]  = yp[1, :]

# Carreau profile per (alpha, frac)
car_eta = {}
car_ux  = {}
for alpha in ALPHAS:
    ds = delta_star_cache[alpha]
    for frac in tqdm(DELTA_FRACS, desc=f'  Carreau profiles alpha={alpha}'):
        delta  = frac * ds
        pd     = ez_numerics.problem_data(alpha, delta, 0)
        sd     = ez_numerics.make_carreau_default_unstable_solution_data()
        eta_c, yc = ez_numerics.compute_carreau_bvp_soln(pd, sd)
        car_eta[(alpha, frac)] = eta_c
        car_ux[(alpha, frac)]  = yc[1, :]


# ── Step 6: Figure 2 — profile grid (3 rows × 4 cols) ────────────────────────
print('\nStep 6: Building new_profiles.png ...')

fig_p, axes_p = plt.subplots(3, 4, figsize=(22, 15))

for (row, alpha) in enumerate(ALPHAS):
    ds       = delta_star_cache[alpha]
    eta_praw = pl_eta_raw[alpha]
    uxp      = pl_ux_raw[alpha]

    for (col_idx, (frac, dlabel, dcol)) in enumerate(
            zip(DELTA_FRACS, DELTA_LABELS, DELTA_COLORS)):
        ax    = axes_p[row, col_idx]
        delta = frac * ds

        # Rescale power-law BVP variable to the physical similarity coordinate eta0:
        #   eta0_pl = eta_pl / delta^((1-alpha)/(2*(1+alpha)))   [mu_inf=0]
        exponent = (1 - alpha) / (2 * (1 + alpha))
        eta0p    = eta_praw / (delta**exponent)

        # Curves
        ax.plot(eta0p,    uxp,                   'r-',
                linewidth=LW, label='Power Law')
        ax.plot(eta_n_p,  ux_n_p,                'k-',
                linewidth=LW, label='Newtonian')
        ax.plot(car_eta[(alpha, frac)], car_ux[(alpha, frac)],
                '--', color=dcol, linewidth=LW, label='Carreau')

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r'$\eta_0$',       fontsize=FS + 1)
        ax.set_ylabel(r'$\bar{u}_x$',    fontsize=FS + 1)
        ax.tick_params(labelsize=FS - 1)

        # Column header: delta label (top row only)
        if row == 0:
            ax.set_title(dlabel, fontsize=FS + 1, pad=8)

        # Row label: alpha (left column only)
        if col_idx == 0:
            ax.set_ylabel(r'$\bar{u}_x$', fontsize=FS + 1)
            ax.annotate(f'$\\alpha = {alpha}$',
                        xy=(0, 0.5), xycoords='axes fraction',
                        xytext=(-0.30, 0.5), textcoords='axes fraction',
                        ha='center', va='center', fontsize=FS + 2,
                        rotation=90,
                        annotation_clip=False)

        ax.legend(fontsize=FS - 3, loc='lower right')

fig_p.tight_layout()
fig_p.savefig('new_profiles.png', dpi=DPI, bbox_inches='tight')
plt.close(fig_p)
print('  Saved new_profiles.png')
print('\nAll done!')
