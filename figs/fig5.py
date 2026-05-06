import ez_numerics
import ez_numerics_blas
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

alphas = np.linspace(.2, 1.6, num=73)
kappa_sak = [ez_numerics.compute_pl_kappa_default(alpha) for alpha in tqdm(alphas)]
kappa_blas = [ez_numerics_blas.compute_pl_kappa_default(alpha) for alpha in tqdm(alphas)]

plt.rcParams.update({'font.size': 14})

plt.plot(alphas, kappa_sak, 'c', label=r'$-\kappa_p$, Sakiadis')
plt.plot(alphas, kappa_blas, 'm', label=r'$\kappa_p$, Blasius')
plt.xlabel(r'$\alpha$', fontsize=30)
# plt.ylabel(r"$\kappa_p$", fontsize=15)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('fig5.eps', format='eps')

