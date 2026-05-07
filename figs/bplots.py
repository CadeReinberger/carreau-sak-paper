import numpy as np
import matplotlib
matplotlib.use('ps')
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from functools import lru_cache
from pathlib import Path

import ez_numerics_blas as ez_numerics


# =========================
# User-editable parameters
# =========================
ALPHAS = [0.3, 1.4]
MU_INF = 0.0

# Easy place to vary the De ranges for each alpha.
DE_VALUES_BY_ALPHA = {
    0.3: np.logspace(0, 2, num=2),
    1.4: np.array([100.0]),
}

# Resolution of the Re_x grid used to draw the tau curves.
NUM_RE_POINTS = 45
RE_PAD_FACTOR = 5.0
ETA_MAX = 10.0
OUTPUT_DIR = Path(__file__).resolve().parent

# Styling
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "legend.framealpha": 1.0,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.6,
})

KAPPA_NEWTONIAN = 0.33205733621519630
DELTA_MULTIPLIERS = [0.1, 1.0, 10.0]
DELTA_COLORS = {
    0.1: "cyan",
    1.0: "tab:green",
    10.0: "tab:red",
}


@lru_cache(maxsize=None)
def pl_kappa(alpha):
    return ez_numerics.compute_pl_kappa_default(alpha)


@lru_cache(maxsize=None)
def carreau_solution(alpha, delta, mu_inf):
    pd = ez_numerics.problem_data(alpha, delta, mu_inf)
    sd = ez_numerics.make_carreau_default_unstable_solution_data()
    return ez_numerics.compute_carreau_bvp_soln(pd, sd)


@lru_cache(maxsize=None)
def carreau_kappa(alpha, delta, mu_inf):
    _, y = carreau_solution(alpha, delta, mu_inf)
    return y[2, 0]


@lru_cache(maxsize=None)
def power_law_profile(alpha):
    sd = ez_numerics.make_pl_default_unstable_solution_data(alpha)
    return ez_numerics.compute_power_law_ivp_soln(alpha, sd)


@lru_cache(maxsize=None)
def newtonian_profile():
    sd = ez_numerics.make_pl_default_unstable_solution_data(1.0)
    return ez_numerics.compute_power_law_ivp_soln(1.0, sd)



def compute_delta_star(alpha):
    kp = pl_kappa(alpha)
    exponent = 2.0 * (1.0 + alpha) / (1.0 - alpha)
    return (kp**alpha / KAPPA_NEWTONIAN) ** exponent



def tau_power_law(alpha, Re_x, De, mu_inf=0.0):
    kp = pl_kappa(alpha)
    prefactor = (1.0 - mu_inf) ** (1.0 / (1.0 + alpha)) if mu_inf < 1.0 else 0.0
    return kp**alpha * (De ** ((alpha - 1.0) / (alpha + 1.0))) * (Re_x ** (-alpha / (1.0 + alpha))) * prefactor



def tau_carreau(alpha, Re_x, De, mu_inf=0.0):
    delta = De**2 / Re_x
    kc = carreau_kappa(alpha, float(delta), mu_inf)
    return kc * Re_x**-0.5 * (mu_inf + (1.0 - mu_inf) * (1.0 + kc**2 * De**2 / Re_x) ** (0.5 * (alpha - 1.0)))



def default_re_grid(alpha, de_values):
    delta_star = compute_delta_star(alpha)
    # Make sure the grid contains the marker locations for the requested delta slices.
    re_candidates = []
    for mult in DELTA_MULTIPLIERS:
        delta = mult * delta_star
        for De in de_values:
            re_candidates.append(De**2 / delta)
    re_candidates = np.array(re_candidates)
    re_min = max(1e-6, np.min(re_candidates) / RE_PAD_FACTOR)
    re_max = np.max(re_candidates) * RE_PAD_FACTOR
    return np.logspace(np.log10(re_min), np.log10(re_max), num=NUM_RE_POINTS)



def build_profile_curves(alpha, delta, mu_inf=0.0):
    etap, yp = power_law_profile(alpha)
    uxp = yp[1, :]
    eta_pl = etap * ((1.0 - mu_inf) ** (1.0 / (alpha + 1.0))) / (delta ** ((1.0 - alpha) / (2.0 + 2.0 * alpha)))

    etan, yn = newtonian_profile()
    ux_n = yn[1, :]

    etac, yc = carreau_solution(alpha, float(delta), mu_inf)
    ux_c = yc[1, :]

    return {
        "power_law": (eta_pl, uxp),
        "newtonian": (etan, ux_n),
        "carreau": (etac, ux_c),
    }



def plot_shear_row():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _mult_to_letter = {10.0: "A", 1.0: "B", 0.1: "C"}

    # plt.rcParams['mathtext.fontset'] = 'cm'

    for subplot_idx, (ax, alpha) in enumerate(zip(axes, ALPHAS), start=1):
        de_values = np.asarray(DE_VALUES_BY_ALPHA[alpha], dtype=float)
        delta_star = compute_delta_star(alpha)
        re_grid = default_re_grid(alpha, de_values)
        if alpha == 0.3:
            re_grid = np.logspace(np.log10(re_grid[0] / 10), np.log10(re_grid[-1] * 10), num=NUM_RE_POINTS)

        tau_newtonian = KAPPA_NEWTONIAN * re_grid**-0.5
        ax.loglog(re_grid, tau_newtonian, color="c", linestyle="--", label="Newtonian")
        all_tau = [tau_newtonian]

        for i_de, De in enumerate(de_values):
            lbl_pl = "Power law" if i_de == 0 else "_nolegend_"
            lbl_c = "Carreau" if i_de == 0 else "_nolegend_"

            tau_pl = tau_power_law(alpha, re_grid, De, MU_INF)
            ax.loglog(re_grid, tau_pl, color="r", linestyle="-.", label=lbl_pl)

            tau_vals = np.array([tau_carreau(alpha, Re_x, De, MU_INF) for Re_x in re_grid])
            # Carreau drawn below other curves (zorder=1)
            ax.loglog(re_grid, tau_vals, color="black", linestyle="-", zorder=1, label=lbl_c)
            all_tau.extend([tau_pl, tau_vals])
            if alpha < 1:
                ax.text(re_grid[2], tau_vals[2] / 1.8, fr"$De={De:g}$",
                        fontsize=9, color="red", ha="left", va="top",
                        bbox=dict(facecolor="white", edgecolor="white", pad=1))

            # Point labels: below markers for alpha<1, above for alpha>1
            y_sign = -1 if alpha < 1 else 1
            prime = "'" if i_de == 1 else ""
            for mult in DELTA_MULTIPLIERS:
                delta = mult * delta_star
                Re_pt = De**2 / delta
                tau_pt = tau_carreau(alpha, Re_pt, De, MU_INF)
                letter = _mult_to_letter[mult]
                pt_label = f"{letter}{subplot_idx}{prime}"

                ax.loglog(
                    [Re_pt], [tau_pt],
                    marker="o", linestyle="None",
                    markersize=9,
                    markerfacecolor=DELTA_COLORS[mult],
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    zorder=7,
                )
                x_off = -8 if i_de == 0 else -8
                trans = mtransforms.offset_copy(ax.transData, fig=fig, x=x_off, y=y_sign * 15, units='points')
                ax.text(Re_pt, tau_pt, pt_label, transform=trans, fontsize=9, ha="center")

        all_tau_flat = np.concatenate(all_tau)
        ax.set_xlim(re_grid[0], re_grid[-1])
        ax.set_ylim(np.min(all_tau_flat) * 1.3, np.max(all_tau_flat) / 1.3)

        ax.set_xlabel(r"$Re_x$")
        ax.set_ylabel(r"$\bar{\tau}_i$" if ax is axes[0] else "")
        ax.set_title(fr"$\alpha={alpha}$")
        ax.legend(loc="best")

    fig.tight_layout()
    outpath = OUTPUT_DIR / "new_shear_blas.eps"
    fig.savefig(outpath, format='eps')
    plt.close(fig)
    return outpath



def plot_profile_grid():
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharex=False, sharey=False)
    _mult_to_letter = {10.0: "A", 1.0: "B", 0.1: "C"}

    for i, alpha in enumerate(ALPHAS):
        subplot_idx = i + 1
        delta_star = compute_delta_star(alpha)
        for j, mult in enumerate(DELTA_MULTIPLIERS):
            delta = mult * delta_star
            ax = axes[i, j]
            curves = build_profile_curves(alpha, delta, MU_INF)

            eta_pl, ux_pl = curves["power_law"]
            eta_n, ux_n = curves["newtonian"]
            eta_c, ux_c = curves["carreau"]

            ax.plot(eta_c, ux_c, color="black", linestyle="-", label="Carreau")
            ax.plot(eta_n, ux_n, color="c", linestyle="--", label="Newtonian")
            ax.plot(eta_pl, ux_pl, color="r", linestyle="-.", label="Power law")

            ax.set_xlim(-0.25, ETA_MAX + 0.25)
            ax.set_ylim(-0.02, 1.03)

            if j == 0:
                ax.set_ylabel(r"$\bar{u}_x$")

            ax.set_title(fr"$\alpha = {alpha},\quad \delta/\delta^* = {mult:g}$", fontsize=20)

            letter = _mult_to_letter[mult]
            pt_label = f"{letter}{subplot_idx}" if subplot_idx == 2 else f"{letter}{subplot_idx}/{letter}{subplot_idx}'"
            ax.plot([0.72], [0.5], marker="o",
                    markerfacecolor=DELTA_COLORS[mult],
                    markeredgecolor="black", markeredgewidth=0.8,
                    markersize=8, transform=ax.transAxes,
                    linestyle="None", clip_on=False, zorder=10)
            ax.text(0.75, 0.5, pt_label, transform=ax.transAxes,
                    ha="left", va="center", fontsize=13, fontweight="bold")

            if i == 0 and j == 0:
                ax.legend(loc="best")

    fig.tight_layout()
    outpath = OUTPUT_DIR / "new_profiles_blas.eps"
    fig.savefig(outpath, format='eps')
    # fig.savefig(outpath, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return outpath



def main():
    shear_path = plot_shear_row()
    profiles_path = plot_profile_grid()
    print(f"Saved shear figure to: {shear_path}")
    print(f"Saved profile figure to: {profiles_path}")


if __name__ == "__main__":
    main()

