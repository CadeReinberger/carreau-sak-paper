import numpy as np
from matplotlib import pyplot as plt
from functools import lru_cache
from pathlib import Path

import ez_numerics


# =========================
# User-editable parameters
# =========================
ALPHAS = [0.3, 0.8, 1.4]
MU_INF = 0.0

# Easy place to vary the De ranges for each alpha.
DE_VALUES_BY_ALPHA = {
    0.3: np.logspace(0, 2, num=2),
    0.8: np.logspace(2, 6, num=2),
    1.4: np.logspace(2, 8, num=2),
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
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.6,
})

KAPPA_NEWTONIAN = 0.4437483133688610511198328438501
DELTA_MULTIPLIERS = [0.01, 0.25, 1.0, 10.0]
DELTA_COLORS = {
    0.01: "tab:blue",
    0.25: "tab:orange",
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
    return -y[2, 0]


@lru_cache(maxsize=None)
def power_law_profile(alpha):
    sd = ez_numerics.make_pl_default_unstable_solution_data(alpha)
    return ez_numerics.compute_power_law_bvp_soln(alpha, sd)


@lru_cache(maxsize=None)
def newtonian_profile():
    sd = ez_numerics.make_pl_default_unstable_solution_data(1.0)
    return ez_numerics.compute_power_law_bvp_soln(1.0, sd)



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
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

    for ax, alpha in zip(axes, ALPHAS):
        de_values = np.asarray(DE_VALUES_BY_ALPHA[alpha], dtype=float)
        delta_star = compute_delta_star(alpha)
        re_grid = default_re_grid(alpha, de_values)

        tau_newtonian = KAPPA_NEWTONIAN * re_grid**-0.5
        ax.loglog(re_grid, tau_newtonian, color="black", linestyle="-", label="Newtonian")

        for De in de_values:
            tau_pl = tau_power_law(alpha, re_grid, De, MU_INF)
            ax.loglog(re_grid, tau_pl, color="c", linestyle="--", label=fr"PL $De={De:g}$")

            tau_vals = np.array([tau_carreau(alpha, Re_x, De, MU_INF) for Re_x in re_grid])
            ax.loglog(re_grid, tau_vals, color="m", linestyle="-.", label=fr"Carreau $De={De:g}$")

            # Marker points corresponding to the requested fixed-delta profiles.
            for mult in DELTA_MULTIPLIERS:
                delta = mult * delta_star
                Re_pt = De**2 / delta
                tau_pt = tau_carreau(alpha, Re_pt, De, MU_INF)
                ax.loglog(
                    [Re_pt],
                    [tau_pt],
                    marker="o",
                    linestyle="None",
                    markersize=9,
                    markerfacecolor=DELTA_COLORS[mult],
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    zorder=5,
                )

        ax.set_xlabel(r"$Re_x$")
        ax.set_ylabel(r"$\tau$" if ax is axes[0] else "")
        ax.set_title(fr"$\alpha={alpha}$")
        ax.grid(True, which="both", alpha=0.25)

        ax.legend(loc="best")

    outpath = OUTPUT_DIR / "new_shear.png"
    fig.savefig(outpath, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return outpath



def plot_profile_grid():
    fig, axes = plt.subplots(3, 4, figsize=(22, 16), sharex=False, sharey=False, constrained_layout=True)

    for i, alpha in enumerate(ALPHAS):
        delta_star = compute_delta_star(alpha)
        for j, mult in enumerate(DELTA_MULTIPLIERS):
            delta = mult * delta_star
            ax = axes[i, j]
            curves = build_profile_curves(alpha, delta, MU_INF)

            eta_pl, ux_pl = curves["power_law"]
            eta_n, ux_n = curves["newtonian"]
            eta_c, ux_c = curves["carreau"]

            ax.plot(eta_n, ux_n, color="black", linestyle="-", label="Newtonian")
            ax.plot(eta_pl, ux_pl, color="0.4", linestyle=":", label="Power law")
            ax.plot(eta_c, ux_c, color=DELTA_COLORS[mult], linestyle="--", label=fr"Carreau, $\delta={mult:g}\delta_*$")

            ax.set_xlim(-0.25, ETA_MAX + 0.25)
            ax.set_ylim(-0.02, 1.03)
            ax.grid(True, alpha=0.25)

            if i == 2:
                ax.set_xlabel(r"$\eta$")
            if j == 0:
                ax.set_ylabel(r"$\bar{u}_x$")

            ax.set_title(fr"$\alpha={alpha},\ \delta={mult:g}\delta_*$")

            if i == 0 and j == 0:
                ax.legend(loc="best")

    outpath = OUTPUT_DIR / "new_profiles.png"
    fig.savefig(outpath, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return outpath



def main():
    shear_path = plot_shear_row()
    profiles_path = plot_profile_grid()
    print(f"Saved shear figure to: {shear_path}")
    print(f"Saved profile figure to: {profiles_path}")


if __name__ == "__main__":
    main()

