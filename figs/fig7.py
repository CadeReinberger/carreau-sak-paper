import numpy as np
import ez_numerics as ez_sak
import ez_numerics_blas as ez_blas
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path


def compute_c(alpha, ez):
    Res = np.logspace(0, 1, num=10)
    De = 1e4

    kappa_p = ez.compute_pl_kappa_default(alpha)

    cs = []
    for Re_x in Res:
        tau_pl = kappa_p**alpha * (De ** ((alpha-1)/(alpha+1))) * (Re_x**(-alpha/(1+alpha)))

        delta = De**2/Re_x
        pd = ez.problem_data(alpha, delta, 0)
        kappa_c = ez.compute_carreau_kappa_default(pd)
        tau_car = kappa_c * Re_x**-.5 * (1 + kappa_c**2*De**2/Re_x)**(.5*(alpha-1))

        c = (tau_pl - tau_car) / tau_car
        cs.append(c)

    return sum(cs) / len(cs)


def plot_fig7():
    alphas = np.linspace(.3, 1.4, num=15)

    print('Computing Sakiadis c(alpha)...')
    cs_sak = -np.array([compute_c(alpha, ez_sak) for alpha in tqdm(alphas)])
    print('Computing Blasius c(alpha)...')
    cs_blas = -np.array([compute_c(alpha, ez_blas) for alpha in tqdm(alphas)])

    alphas = list(alphas)
    cs_sak = list(cs_sak)
    cs_blas = list(cs_blas)
    del alphas[4]
    del cs_sak[4]
    del cs_blas[4]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, cs_sak, linewidth=2.5, color='cyan', label='Sakiadis', zorder=2)
    ax.plot(alphas, cs_blas, linewidth=2.5, color='magenta', linestyle='-.', label='Blasius', zorder=3)
    ax.set_xlabel(r'$\alpha$', fontsize=20)
    ax.set_ylabel(r'$c(\alpha)$', fontsize=20)
    ax.tick_params(labelsize=15)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=14)
    fig.tight_layout()

    outpath = Path(__file__).resolve().parent / 'fig7.png'
    fig.savefig(outpath, dpi=350, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved to {outpath}')


plot_fig7()
