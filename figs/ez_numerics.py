import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_bvp
from matplotlib import pyplot as plt

''' First, some structure to deal with how we want to solve '''

@dataclass 
class problem_data:
    alpha: float
    delta: float
    mu_inf: float
    
def unpack(pd):
    return (pd.alpha, pd.delta, pd.mu_inf)
    
@dataclass
class solution_data:
    is_stable: bool
    tol: float 
    l: float
    max_nodes: float
    init_nodes: float
    mesh_inner_length_ratio: float
    mesh_inner_density_fraction: float
    
def make_carreau_default_unstable_solution_data():
    return solution_data(False, 1e-8, 50, 1e5, 1250, .1, .6)

def make_pl_default_unstable_solution_data(alpha):
    if .5 < alpha < 1:
        return solution_data(False, 1e-4, 50, 1e5, 1000, .1, .5)
    elif alpha < .5:
        return solution_data(False, 1e-4, 50, 1e5, 1250, .2, .5)
    else: 
        return solution_data(False, 1e-4, 20, 1e5, 1500, .075, .64)

    
    
''' Next, the structures to actually just solve the problem '''
    

def compute_carreau_bvp_soln(pd, sd):
    #  Eventually, this will be beyond numerical reproach. 
    if sd.is_stable: 
        raise Exception('Fully Stable Solution is Not Yet Implemented. Please write some fucking matlab you fucking shit-for-brains.')
    alpha, delta, mu_inf = unpack(pd) # unpack and get the problem data
    # Make the BVP itself
    def f(x, y):
        f, fp, fpp = y[0], y[1], y[2]
        fppp = -f*fpp/(2*(mu_inf + (1-mu_inf)*(1+alpha*delta*fpp**2)*(1+delta*fpp**2)**(.5*(alpha-3)) ))
        return np.vstack((fp, fpp, fppp))
    def bc(ya, yb):
        return np.array([ya[0], ya[1]-1, yb[1]])
    # Make the Mesh to use as an initial guess
    L1 = sd.mesh_inner_length_ratio * sd.l
    L2 = sd.l 
    num1 = int(sd.init_nodes * sd.mesh_inner_density_fraction)
    num2 = int(sd.init_nodes * (1-sd.mesh_inner_density_fraction))
    small_delta = min(L1/(num1-1), (L2-L1)/(num2-1))
    x = np.hstack((np.linspace(0, L1, num=num1), np.linspace(L1+small_delta, L2, num=num2)))
    # Fill the mesh with the first two asymptiotic terms of the Newtonian
    C = 1.616125446804603717027117425028
    G = -2.1313459240475714821
    guess_f = C + G*np.exp(-.5*C*x)
    guess_fp = -.5*C*guess_f*G*np.exp(-.5*C*x)
    guess_fpp = .25*C*C*G*np.exp(-.5*C*x)
    y = np.vstack((guess_f, guess_fp, guess_fpp))
    # solve that thing
    bvp_soln = solve_bvp(f, bc, x, y, tol=sd.tol, max_nodes=sd.max_nodes)
    return (bvp_soln.x, bvp_soln.y)

def compute_carreau_kappa_default(pd):
    sd = make_carreau_default_unstable_solution_data()
    x, y = compute_carreau_bvp_soln(pd, sd)
    return -y[2,0]

def compute_power_law_bvp_soln(alpha, sd):
    #  Eventually, this will be beyond numerical reproach. 
    if sd.is_stable: 
        raise Exception('Fully Stable Solution is Not Yet Implemented. Please write some fucking matlab you fucking shit-for-brains.')
    # Make the BVP itself
    def f(x, y):
        f, fp, fpp = y[0], y[1], y[2]
        fppp = -(f*fpp*(np.abs(fpp)**(1-alpha)))/(alpha*(alpha+1))
        return np.vstack((fp, fpp, fppp))
    def bc(ya, yb):
        return np.array([ya[0], ya[1]-1, yb[1]])
    # Make the Mesh to use as an initial guess
    L1 = sd.mesh_inner_length_ratio * sd.l
    L2 = sd.l 
    num1 = int(sd.init_nodes * sd.mesh_inner_density_fraction)
    num2 = int(sd.init_nodes * (1-sd.mesh_inner_density_fraction))
    small_delta = min(L1/(num1-1), (L2-L1)/(num2-1))
    x = np.hstack((np.linspace(0, L1, num=num1), np.linspace(L1+small_delta, L2, num=num2)))
    # Fill the mesh with the first two asymptiotic terms of the Newtonian
    C = 1.616125446804603717027117425028
    G = -2.1313459240475714821
    guess_f = C + G*np.exp(-.5*C*x)
    guess_fp = -.5*C*guess_f*G*np.exp(-.5*C*x)
    guess_fpp = .25*C*C*G*np.exp(-.5*C*x)
    y = np.vstack((guess_f, guess_fp, guess_fpp))
    # solve that thing
    bvp_soln = solve_bvp(f, bc, x, y, tol=sd.tol, max_nodes=sd.max_nodes)
    return (bvp_soln.x, bvp_soln.y)


def compute_pl_kappa_default(alpha):
    sd = make_pl_default_unstable_solution_data(alpha)
    x, y = compute_power_law_bvp_soln(alpha, sd)
    return -y[2,0]
