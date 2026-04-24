import numpy  as np
import ez_numerics
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.cm as colormap

''' 
These are all designed as helper scripts, so change the top lines of each
function to change the plots
'''

def plot_tau_vs_rex_no_mu_inf(alpha):
    
    # The main plotting params and such
    Res = np.logspace(-1, 8, num=81)
    Des = np.logspace(2, 4, num=3)
    
    # Now, we simply get Newtonian shear plot
    kappa_n = 0.4437483133688610511198328438501
    tau_n = kappa_n * Res**-.5
    
    plt.loglog(Res, tau_n, 'k-', label='Newtonian')
    
    # Next, precompute the power-law shear
    kappa_p = ez_numerics.compute_pl_kappa_default(alpha)
    
    # Make the background prediction plot
    # plt.loglog(Res, .1*Res**(-alpha/(1+alpha)), 'k-.', label='Slope -a/(a+1)')
    # plt.loglog(Res, .1*Res**(-1-alpha/(1+alpha)), 'k-.', label='Slope -a/(a+1)')

    
    # Now, we iterate over the Deborah numbers to get what we want
    for De in Des:
        
        # First, compute the power law shear that way
        tau_pl = kappa_p**alpha * (De ** ((alpha-1)/(alpha+1))) * (Res**(-alpha/(1+alpha)))
        
        # Initialize tau for carreau
        tau_car = np.zeros(len(Res))
        
        # Second, compute the power law shear that way
        for (ind, Re) in enumerate(tqdm(Res)):
            delta = De**2/Re
            pd = ez_numerics.problem_data(alpha, delta, 0) # because no mu_inf
            kappa_c = ez_numerics.compute_carreau_kappa_default(pd)
            tau = kappa_c * Re**-.5 * (1 + kappa_c**2*De**2/Re)**(.5*(alpha-1))
            tau_car[ind] = tau
        
        # Finally add it to the plot
        plt.semilogx(Res, tau_pl, 'c--', label=f'PL De={De}')
        plt.semilogx(Res, tau_car, 'm-.', label=f'Carreau De={De}')
        # plt.semilogx(Res, np.abs((tau_pl-tau_car) / (tau_car)), '--', label=f'De={De}')
        
    plt.xlabel('Re_x')
    plt.ylabel('Relative Error in Tau')
    plt.title(f'Dimensionless Wall Shear for Carreau with alpha={alpha}')
    plt.legend(loc=(1.05, 0.25))
    # plt.ylim([0, 1])
    plt.show()
    
    
def compute_c(alpha):
    # Just use like a fixed De and Re_x. 
    Res = np.logspace(0, 1, num=10)
    De = 1e4
    
    # First, do the power-law computation
    kappa_p = ez_numerics.compute_pl_kappa_default(alpha)
    
    # Next, do the relevant Carreau computation
    cs = []
    for Re_x in Res:
        tau_pl = kappa_p**alpha * (De ** ((alpha-1)/(alpha+1))) * (Re_x**(-alpha/(1+alpha)))
        
        delta = De**2/Re_x
        pd = ez_numerics.problem_data(alpha, delta, 0) # because no mu_inf
        kappa_c = ez_numerics.compute_carreau_kappa_default(pd)
        tau_car = kappa_c * Re_x**-.5 * (1 + kappa_c**2*De**2/Re_x)**(.5*(alpha-1))
    
        c = np.abs((tau_pl - tau_car) / tau_car)
        cs.append(c)
        
    c_avg = sum(cs) / len(cs)
    return c_avg
    
def plot_tau_vs_rex_with_mu_inf(alpha, mu_inf_bar):
    
    # The main plotting params and such
    Res = np.logspace(-1, 8, num=81)
    Des = np.logspace(2, 6, num=3)
    
    # Now, we simply get Newtonian shear plot
    kappa_n = 0.4437483133688610511198328438501
    tau_n_0 = kappa_n * Res**-.5
    
    plt.loglog(Res, tau_n_0, 'k-', label='Newtonian Mu_0')
    
    # Now we add in the Newtonian shear plot with the smaller viscosity
    kappa_n = 0.4437483133688610511198328438501
    tau_n_inf = kappa_n * np.sqrt(mu_inf_bar) * Res**-.5
    
    plt.loglog(Res, tau_n_inf, 'k-', label='Newtonian Mu_Inf')
    
    # Next, precompute the power-law shear
    kappa_p = ez_numerics.compute_pl_kappa_default(alpha)
    
    # Now, we iterate over the Deborah numbers to get what we want
    for De in Des:
        
        # First, compute the power law shear that way
        tau_pl = kappa_p**alpha * (De ** ((alpha-1)/(alpha+1))) * (Res**(-alpha/(1+alpha))) * (1-mu_inf_bar)**(1/(1+alpha))
        
        # Initialize tau for carreau
        tau_car = np.zeros(len(Res))
        
        # Second, compute the power law shear that way
        for (ind, Re) in enumerate(tqdm(Res)):
            delta = De**2/Re
            pd = ez_numerics.problem_data(alpha, delta, mu_inf_bar)
            kappa_c = ez_numerics.compute_carreau_kappa_default(pd)
            tau = kappa_c * Re**-.5 * (mu_inf_bar + (1-mu_inf_bar)*(1 + kappa_c**2*De**2/Re)**(.5*(alpha-1)))
            tau_car[ind] = tau
        
        # Finally add it to the plot
        plt.loglog(Res, tau_pl, 'c--', label=f'PL De={De}')
        plt.loglog(Res, tau_car, 'm-.', label=f'Carreau De={De}')
        
    plt.xlabel('Re_x')
    plt.ylabel('tau')
    plt.title(f'Dimensionless Wall Shear for Carreau with alpha={alpha} mu_inf_bar={mu_inf_bar}')
    plt.legend(loc=(1.05, 0.25))
    plt.show()
    
def plot_pl_kappa():
    alphas = np.linspace(.2, 1.8, num=73)
    kappas = [ez_numerics.compute_pl_kappa_default(alpha) for alpha in tqdm(alphas)]
    
    plt.plot(alphas, kappas)
    plt.xlabel('alapha')
    plt.ylabel('kappa')
    plt.show()

def plot_carreau_kappa():
    alphas = np.linspace(.6, 1.4, num=5)
    deltas = np.logspace(-2, 6, num=100)
    for alpha in alphas:
        kappas = [ez_numerics.compute_carreau_kappa_default(ez_numerics.problem_data(alpha, delta, 0)) for delta in tqdm(deltas)]
        plt.semilogx(deltas, kappas, label=f'alpha={alpha}')
    plt.xlabel('De^2/Re_x')
    plt.ylabel('kappa')
    plt.legend()
    plt.show()
    
def make_shear_phase_plot_with_mu_inf_basic_once(alpha):
    # start by getting delta_star_0
    kappa_n = 0.4437483133688610511198328438501
    kappa_p = ez_numerics.compute_pl_kappa_default(alpha)
    delta_star_0 = (kappa_p**alpha/kappa_n) ** (2*(1+alpha)/(1-alpha))
    
    # get the mu_inf_bar to check
    mu_inf_bar = np.logspace(-3, 0, num=50)
    
    # Now, compute the high cutoff
    delta_star_low = delta_star_0 * (1-mu_inf_bar)**(2/(1-alpha))
    
    # Now, the low cutoff
    delta_star_high = delta_star_low * mu_inf_bar ** ((alpha+1)/(alpha-1))
    
    # Add those plots
    plt.loglog(mu_inf_bar, delta_star_low, 'r-')
    plt.loglog(mu_inf_bar, delta_star_high, 'r-')
    
    # Show it 
    plt.xlabel('mu_inf_bar')
    plt.ylabel('De^2/Re_x')
    plt.title(f'Truncated Power Law Regions for alpha={alpha}')
    plt.show()
        
def make_shear_phase_plot_with_mu_inf_bougie_once(alpha):
    # start by getting delta_star_0
    kappa_n = 0.4437483133688610511198328438501
    kappa_p = ez_numerics.compute_pl_kappa_default(alpha)
    delta_star_0 = (kappa_p**alpha/kappa_n) ** (2*(1+alpha)/(1-alpha))
    
    # get the mu_inf_bar to check
    mu_inf_bar = np.logspace(-3, 0, num=50)
    
    # Now, compute the high cutoff
    delta_star_low = delta_star_0 * (1-mu_inf_bar)**(2/(1-alpha))
    
    # Now, the low cutoff
    delta_star_high = delta_star_low * mu_inf_bar ** ((alpha+1)/(alpha-1))
    
    # Now prepare to hack contourf to make this work
    deltas = np.logspace(-5, 20, num=100)
    M, D = np.meshgrid(mu_inf_bar, deltas)
    zs = np.zeros(np.shape(M))
    
    # Fill the array appropriately
    for (m_ind, m) in enumerate(mu_inf_bar):
        for (d_ind, d) in enumerate(deltas):
            col = 0 # Start low shear Newtonian 
            if d > delta_star_high[m_ind]:
                col = 2 # High shear newtonian
            elif d > delta_star_low[m_ind]:
                col = 1 # power law region
            zs[d_ind, m_ind] = col
    
    # Then we contourf it
    plt.contourf(M, D, zs)
    
    plt.xscale('log')
    plt.yscale('log')
    
    # Show it 
    plt.xlabel('mu_inf_bar')
    plt.ylabel('De^2/Re_x')
    plt.title(f'Truncated Power Law Regions for alpha={alpha}')
    plt.show()
    
    
def make_shear_phase_plot_with_mu_inf_many():
    # make list of alphas
    alphas = np.linspace(.2, .8, num=7)
    
    cm = colormap.get_cmap("plasma") 
    
    for (ind, alpha) in enumerate(alphas):
    
        # start by getting delta_star_0
        kappa_n = 0.4437483133688610511198328438501
        kappa_p = ez_numerics.compute_pl_kappa_default(alpha)
        delta_star_0 = (kappa_p**alpha/kappa_n) ** (2*(1+alpha)/(1-alpha))
        
        # get the mu_inf_bar to check
        mu_inf_bar = np.logspace(-3, 0, num=50)
        
        # Now, compute the high cutoff
        delta_star_low = delta_star_0 * (1-mu_inf_bar)**(2/(1-alpha))
        
        # Now, the low cutoff
        delta_star_high = delta_star_low * mu_inf_bar ** ((alpha+1)/(alpha-1))
        
        # Add those plots
        col = cm(ind/len(alphas))
        plt.loglog(mu_inf_bar, delta_star_low, '-.', alpha=.5, color=col, label=f'alpha={round(alpha,1)}')
        plt.loglog(mu_inf_bar, delta_star_high, '--', alpha=.5, color=col)
    
    # Show it 
    plt.xlabel('mu_inf_bar')
    plt.ylabel('De^2/Re_x')
    plt.legend()
    plt.title('Truncated Power Law Regions for Wall Shear')
    plt.show()
    
def compute_suite_of_profiles_hakuna_matata(pd, eta_max=10):

    # Now, let's get the Power law solution 
    etap, y = ez_numerics.compute_power_law_bvp_soln(pd.alpha, ez_numerics.make_pl_default_unstable_solution_data(pd.alpha))
    uxp = y[1,:]
    eta0p = etap * ((1-pd.mu_inf)**(1/(pd.alpha+1))) / (pd.delta**((1-pd.alpha)/(2+2*pd.alpha)))
    plt.plot(eta0p, uxp, 'r-', label='Power Law')
    
    # Now we add in Newtonian0
    eta00, y = ez_numerics.compute_power_law_bvp_soln(1, ez_numerics.make_pl_default_unstable_solution_data(1))
    uxn0 = y[1,:]
    plt.plot(eta00, uxn0, 'k-', label='mu0')
    
    # If needed, we add in Newtonian with mu infinity
    if pd.mu_inf > 0:
        etaninf = np.sqrt(pd.mu_inf) * eta00
        plt.plot(etaninf, uxn0, 'b-', label='mu inf')
    
    # First, let's get the Carreau solution to return 
    sd = ez_numerics.make_carreau_default_unstable_solution_data()
    eta0, y = ez_numerics.compute_carreau_bvp_soln(pd, sd)
    ux = y[1,:]
    plt.plot(eta0, ux, 'g--', label='Carreau')
    
    plt.xlabel('eta')
    plt.ylabel('ux_bar')
    plt.title(f'Profiles for alpha={pd.alpha}, delta={pd.delta}' + (f', mu_inf_bar={pd.mu_inf}' if pd.mu_inf > 0 else ''))
    plt.xlim((-.5, eta_max+.5))
    plt.legend()
    plt.show()
    
def compute_suite_of_profiles_hakuna_matata_transposed(pd, eta_max=10):
    
    # TODO: ADD in shears?
    # TODO: Is there more we can say about shear locality? 

    # Now, let's get the Power law solution 
    etap, y = ez_numerics.compute_power_law_bvp_soln(pd.alpha, ez_numerics.make_pl_default_unstable_solution_data(pd.alpha))
    uxp = y[1,:]
    eta0p = etap * ((1-pd.mu_inf)**(1/(pd.alpha+1))) / (pd.delta**((1-pd.alpha)/(2+2*pd.alpha)))
    plt.plot(uxp, eta0p, 'r-', label='Power Law')
    
    # Now we add in Newtonian0
    eta00, y = ez_numerics.compute_power_law_bvp_soln(1, ez_numerics.make_pl_default_unstable_solution_data(1))
    uxn0 = y[1,:]
    plt.plot(uxn0, eta00, 'k-', label='mu0')
    
    # If needed, we add in Newtonian with mu infinity
    if pd.mu_inf > 0:
        etaninf = np.sqrt(pd.mu_inf) * eta00
        plt.plot(uxn0, etaninf, 'b-', label='mu inf')
    
    # First, let's get the Carreau solution to return 
    sd = ez_numerics.make_carreau_default_unstable_solution_data()
    eta0, y = ez_numerics.compute_carreau_bvp_soln(pd, sd)
    ux = y[1,:]
    plt.plot(ux, eta0, 'g--', label='Carreau')
    
    plt.xlabel('ux_bar')
    plt.ylabel('eta')
    plt.title(f'Profiles for alpha={pd.alpha}, delta={pd.delta}' + (f', mu_inf_bar={pd.mu_inf}' if pd.mu_inf > 0 else ''))
    plt.ylim((-.5, eta_max+.5))
    plt.legend()
    plt.show()

    
def make_full_approximate_hakuna_matata_plots(trans=False):
    if not trans:
        compute_suite_of_profiles_hakuna_matata(ez_numerics.problem_data(.8, 50, 0))    
        compute_suite_of_profiles_hakuna_matata(ez_numerics.problem_data(.8, 1e16, .1))
        compute_suite_of_profiles_hakuna_matata(ez_numerics.problem_data(.3, 5, 0))
        compute_suite_of_profiles_hakuna_matata(ez_numerics.problem_data(.3, 250, 0))
        compute_suite_of_profiles_hakuna_matata(ez_numerics.problem_data(1.5, 30, 0))
        compute_suite_of_profiles_hakuna_matata(ez_numerics.problem_data(1.5, 5, 0))
    else:
        compute_suite_of_profiles_hakuna_matata_transposed(ez_numerics.problem_data(.8, 50, 0))    
        compute_suite_of_profiles_hakuna_matata_transposed(ez_numerics.problem_data(.8, 1e16, .1))
        compute_suite_of_profiles_hakuna_matata_transposed(ez_numerics.problem_data(.3, 5, 0))
        compute_suite_of_profiles_hakuna_matata_transposed(ez_numerics.problem_data(.3, 250, 0))
        compute_suite_of_profiles_hakuna_matata_transposed(ez_numerics.problem_data(1.5, 30, 0))
        compute_suite_of_profiles_hakuna_matata_transposed(ez_numerics.problem_data(1.5, 5, 0))
        
        
        
# compute_suite_of_profiles_hakuna_matata_transposed(ez_numerics.problem_data(1.5, 30, 0))  
# compute_suite_of_profiles_hakuna_matata_transposed(ez_numerics.problem_data(1.5, 5, 0))

# mak
        
# plot_tau_vs_rex_no_mu_inf(1.3)
# make_shear_phase_plot_with_mu_inf_many()

# make_full_approximate_hakuna_matata_plots(True)
# 

# for delta in range(10, 210, 20):
#     compute_suite_of_profiles_hakuna_matata_transposed(ez_numerics.problem_data(1.5, delta, 0))

def plot_c():
    alphas = np.linspace(.3, 1, num=15)
    cs = np.array([compute_c(alpha) for alpha in tqdm(alphas)])
    alphas, cs = list(alphas), list(cs)
    del(alphas[4])
    del(cs[4])
    plt.plot(alphas, cs)
    plt.xlabel('alpha')
    plt.ylabel('c(alpha)')
    plt.show()
    
plot_c()





    
    
    
    
