import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_bvp, solve_ivp


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
    return solution_data(False, 1e-4, 20, 1e5, 1500, .075, .64)

def make_pl_default_unstable_solution_data(alpha):
    if .5 < alpha < 1:
        return solution_data(False, 1e-4, 50, 1e5, 1000, .1, .5)
    elif alpha < .5:
        return solution_data(False, 1e-4, 50, 1e5, 1250, .2, .5)
    else: 
        return solution_data(False, 1e-3, 20, 1e6, 3000, .1, .6)

    
    
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
        return np.array([ya[0], ya[1], yb[1]-1])
    # Make the Mesh to use as an initial guess
    L1 = sd.mesh_inner_length_ratio * sd.l
    L2 = sd.l 
    num1 = int(sd.init_nodes * sd.mesh_inner_density_fraction)
    num2 = int(sd.init_nodes * (1-sd.mesh_inner_density_fraction))
    small_delta = min(L1/(num1-1), (L2-L1)/(num2-1))
    x = np.hstack((np.linspace(0, L1, num=num1), np.linspace(L1+small_delta, L2, num=num2)))
    # Fill the mesh with the first two asymptiotic terms of the Newtonian
    guess_f = x-1.7
    guess_fp = np.ones(x.shape)
    guess_fpp = np.zeros(x.shape)
    y = np.vstack((guess_f, guess_fp, guess_fpp))
    # solve that thing
    bvp_soln = solve_bvp(f, bc, x, y, tol=sd.tol, max_nodes=sd.max_nodes)
    return (bvp_soln.x, bvp_soln.y)

def compute_carreau_kappa_default(pd):
    sd = make_carreau_default_unstable_solution_data()
    x, y = compute_carreau_bvp_soln(pd, sd)
    return y[2,0]

# def compute_power_law_bvp_soln(alpha, sd):
#     #  Eventually, this will be beyond numerical reproach. 
#     if sd.is_stable: 
#         raise Exception('Fully Stable Solution is Not Yet Implemented. Please write some fucking matlab you fucking shit-for-brains.')
#     # Make the BVP itself
#     def f(x, y):
#         f, fp, fpp = y[0], y[1], y[2]
#         fppp = -(f*fpp*(np.abs(fpp)**(1-alpha)))/(alpha*(alpha+1))
#         return np.vstack((fp, fpp, fppp))
#     def bc(ya, yb):
#         return np.array([ya[0], ya[1], yb[1]-1])
#     # Make the Mesh to use as an initial guess
#     L1 = sd.mesh_inner_length_ratio * sd.l
#     L2 = sd.l 
#     num1 = int(sd.init_nodes * sd.mesh_inner_density_fraction)
#     num2 = int(sd.init_nodes * (1-sd.mesh_inner_density_fraction))
#     small_delta = min(L1/(num1-1), (L2-L1)/(num2-1))
#     x = np.hstack((np.linspace(0, L1, num=num1), np.linspace(L1+small_delta, L2, num=num2)))
#     # Fill the mesh with the first two asymptiotic terms of the Newtonian
#     guess_f = x-1.7
#     guess_fp = np.ones(x.shape)
#     guess_fpp = np.zeros(x.shape)
#     y = np.vstack((guess_f, guess_fp, guess_fpp))
#     # solve that thing
#     bvp_soln = solve_bvp(f, bc, x, y, tol=sd.tol, max_nodes=sd.max_nodes)
#     return (bvp_soln.x, bvp_soln.y)

def compute_power_law_ivp_soln(alpha, sd, s0=1.0):
    """
    IVP version of the power-law Blasius-like BVP using scaling symmetry.

    Solves on a computational variable xi with IVP:
        f(0)=0, f'(0)=0, f''(0)=s0
    then rescales (xi,f) -> (x, f_tilde) so that f_tilde'(infty)=1.

    Returns (x, y) where y = [f, f', f''] sampled on x in [0, sd.l].
    """
    if sd.is_stable:
        raise Exception('Fully Stable Solution is Not Yet Implemented. Please write some fucking matlab you fucking shit-for-brains.')

    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if np.isclose(alpha, 2.0):
        raise ValueError("alpha = 2 makes the scaling exponent singular in this formulation.")

    # --- ODE in xi-variable ---
    # y = [f, fp, fpp]
    def rhs(xi, y):
        f, fp, fpp = y
        fppp = -(f * fpp * (np.abs(fpp) ** (1 - alpha))) / (alpha * (alpha + 1))
        return np.array([fp, fpp, fppp], dtype=float)

    y0 = np.array([0.0, 0.0, float(s0)], dtype=float)

    # --- Scaling law derivation (1-parameter invariance) ---
    # If f(x) solves, then f_tilde(x) = b f(a x) solves provided:
    #     b = a^p,  p = (2α - 1)/(2 - α)
    # Then f_tilde'(∞) = (b a) f'(∞) = a^(p+1) f'(∞)
    # Let k = p+1 = (α+1)/(2-α)
    # Choose a so that a^k * U = 1  =>  a = U^(-(2-α)/(α+1))
    p = (2 * alpha - 1) / (2 - alpha)
    k = (alpha + 1) / (2 - alpha)
    a_power = - (2 - alpha) / (alpha + 1)  # exponent on U: a = U^a_power

    def integrate_to(xi_max):
        sol = solve_ivp(
            rhs,
            t_span=(0.0, float(xi_max)),
            y0=y0,
            method=getattr(sd, "ivp_method", "RK45"),
            rtol=sd.tol,
            atol=getattr(sd, "atol", sd.tol * 1e-3),
            dense_output=True,
            max_step=getattr(sd, "ivp_max_step", np.inf),
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return sol

    # Fixed-point on xi_max so final x-range is [0, sd.l]
    # We need sol defined up to xi = a * sd.l, but a depends on U=f'(xi_max).
    xi_max = float(sd.l)
    for _ in range(3):
        sol = integrate_to(xi_max)
        U = float(sol.y[1, -1])  # fp at xi_max
        if U <= 0:
            raise RuntimeError(f"Nonpositive far-field slope encountered (U={U}). Try different s0/method/xi_max.")
        a = U ** a_power
        xi_max_new = a * float(sd.l)
        if xi_max_new <= 0:
            raise RuntimeError("Computed nonpositive xi_max after scaling; something went off the rails.")
        if abs(xi_max_new - xi_max) / max(xi_max, 1.0) < 1e-3:
            xi_max = xi_max_new
            break
        xi_max = xi_max_new

    # Final integration to the needed xi range
    sol = integrate_to(xi_max)
    U = float(sol.y[1, -1])
    if U <= 0:
        raise RuntimeError(f"Nonpositive far-field slope encountered (U={U}) on final run.")

    a = U ** a_power
    b = a ** p

    # Sample on the *physical* x-grid exactly like your BVP code did (same sd.l scale)
    L = float(sd.l)
    x = np.linspace(0.0, L, num=int(sd.init_nodes))

    # Map to xi = a x and evaluate base solution there
    xi = a * x
    Y = sol.sol(xi)  # shape (3, N), base [f, fp, fpp] in xi-variable

    f_base, fp_base, fpp_base = Y[0], Y[1], Y[2]

    # Rescale into the normalized solution on x-variable
    f_out   = b * f_base
    fp_out  = (b * a) * fp_base
    fpp_out = (b * a**2) * fpp_base

    y_out = np.vstack((f_out, fp_out, fpp_out))

    return x, y_out


def compute_power_law_soln_forward_stepping(alpha, sd):
    #  Eventually, this will be beyond numerical reproach. 
    if sd.is_stable: 
        raise Exception('Fully Stable Solution is Not Yet Implemented. Please write some fucking matlab you fucking shit-for-brains.')
    # Make the BVP itself
    def f(x, y):
        f, fp, fpp = y[0], y[1], y[2]
        fppp = -(f*fpp*(np.abs(fpp)**(1-alpha)))/(alpha*(alpha+1))
        return np.vstack((fp, fpp, fppp))
    


def compute_pl_kappa_default(alpha):
    sd = make_pl_default_unstable_solution_data(alpha)
    x, y = compute_power_law_ivp_soln(alpha, sd)
    return y[2,0]
