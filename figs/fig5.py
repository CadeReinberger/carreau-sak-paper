import ez_numerics
import ez_numerics_blas
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

alphas = np.linspace(.2, 1.6, num=73)
kappa_sak = [ez_numerics.compute_pl_kappa_default(alpha) for alpha in tqdm(alphas)]
kappa_blas = [ez_numerics_blas.compute_pl_kappa_default(alpha) for alpha in tqdm(alphas)]

plt.rcParams.update({'font.size': 14})

plt.plot(alphas, kappa_sak, 'c', label='Sakiadis')
plt.plot(alphas, kappa_blas, 'm', label='Blasius')
plt.xlabel(r'$\alpha$', fontsize=15)
plt.ylabel(r"$|f_p''(0;\,\alpha)|$", fontsize=15)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig('fig5.png', dpi=250)

