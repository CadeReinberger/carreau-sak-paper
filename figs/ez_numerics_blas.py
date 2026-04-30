import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from functools import lru_cache
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


@lru_cache(maxsize=None)
def _pl_warm_start(alpha):
    """Power-law IVP solution cached per alpha, used as BVP warm start for large delta."""
    sd = make_pl_default_unstable_solution_data(alpha)
    return compute_power_law_ivp_soln(alpha, sd)


def _carreau_bc(ya, yb):
    return np.array([ya[0], ya[1], yb[1] - 1])


def _carreau_mesh(sd):
    L1 = sd.mesh_inner_length_ratio * sd.l
    L2 = sd.l
    num1 = int(sd.init_nodes * sd.mesh_inner_density_fraction)
    num2 = int(sd.init_nodes * (1 - sd.mesh_inner_density_fraction))
    eps = min(L1 / (num1 - 1), (L2 - L1) / (num2 - 1))
    return np.hstack((np.linspace(0, L1, num=num1),
                      np.linspace(L1 + eps, L2, num=num2)))


def _interp_onto(x_new, x_old, y_old):
    out = np.zeros((y_old.shape[0], len(x_new)))
    for i in range(y_old.shape[0]):
        out[i] = np.interp(x_new, x_old, y_old[i], right=y_old[i, -1])
    return out


# Shared cache: (alpha, delta, mu_inf) → (x, y).  Used both for the backbone
# and for every regular solve so the backbone solutions are reused across calls.
_carreau_bvp_cache = {}

# Step ratio for the backbone ladder (powers of _BACKBONE_STEP starting at 1).
_BACKBONE_STEP = 5.0


def _backbone_floor(delta):
    """Largest backbone delta (power of _BACKBONE_STEP, >= 1) that does not exceed delta."""
    d = 1.0
    while d * _BACKBONE_STEP <= delta:
        d *= _BACKBONE_STEP
    return d


def _ensure_backbone(alpha, mu_inf, delta_max, sd):
    """
    Walk up the backbone ladder 1, 5, 25, … caching BVP solutions as we go.
    Stops at the largest backbone delta that does not exceed delta_max.
    Each step uses the previous backbone solution as warm start, so every
    solve is a ≤5× continuation step.
    """
    x0 = _carreau_mesh(sd)
    d = 1.0
    x_prev, y_prev = None, None

    while True:
        key = (alpha, d, mu_inf)
        if key not in _carreau_bvp_cache:
            if x_prev is None:
                x_pl, y_pl = _pl_warm_start(alpha)
                y_init = _interp_onto(x0, x_pl, y_pl)
            else:
                y_init = _interp_onto(x0, x_prev, y_prev)

            def f_bb(x, y, _d=d):
                fv, fp, fpp = y[0], y[1], y[2]
                fppp = -fv*fpp / (2*(mu_inf + (1-mu_inf)*(1+alpha*_d*fpp**2)*(1+_d*fpp**2)**(.5*(alpha-3))))
                return np.vstack((fp, fpp, fppp))

            sol = solve_bvp(f_bb, _carreau_bc, x0.copy(), y_init,
                            tol=sd.tol, max_nodes=sd.max_nodes)
            if not sol.success:
                raise RuntimeError(
                    f"Backbone BVP failed (alpha={alpha}, delta={d:.3g}): {sol.message}")
            _carreau_bvp_cache[key] = (sol.x, sol.y)

        x_prev, y_prev = _carreau_bvp_cache[key]

        if d * _BACKBONE_STEP > delta_max:
            break
        d *= _BACKBONE_STEP


def compute_carreau_bvp_soln(pd, sd):
    #  Eventually, this will be beyond numerical reproach.
    if sd.is_stable:
        raise Exception('Fully Stable Solution is Not Yet Implemented. Please write some fucking matlab you fucking shit-for-brains.')
    alpha, delta, mu_inf = unpack(pd)

    cache_key = (alpha, delta, mu_inf)
    if cache_key in _carreau_bvp_cache:
        return _carreau_bvp_cache[cache_key]

    def f_ode(x, y):
        fv, fp, fpp = y[0], y[1], y[2]
        fppp = -fv*fpp / (2*(mu_inf + (1-mu_inf)*(1+alpha*delta*fpp**2)*(1+delta*fpp**2)**(.5*(alpha-3))))
        return np.vstack((fp, fpp, fppp))

    x = _carreau_mesh(sd)

    if delta < 1.0:
        y_init = np.vstack((x - 1.7, np.ones(x.shape), np.zeros(x.shape)))
    elif delta < _BACKBONE_STEP:
        x_pl, y_pl = _pl_warm_start(alpha)
        y_init = _interp_onto(x, x_pl, y_pl)
    else:
        # Build the backbone up to the floor of delta, then warm-start from there.
        _ensure_backbone(alpha, mu_inf, delta, sd)
        d_floor = _backbone_floor(delta)
        x_bb, y_bb = _carreau_bvp_cache[(alpha, d_floor, mu_inf)]
        y_init = _interp_onto(x, x_bb, y_bb)

    sol = solve_bvp(f_ode, _carreau_bc, x, y_init, tol=sd.tol, max_nodes=sd.max_nodes)
    if not sol.success:
        raise RuntimeError(f"Carreau BVP failed (delta={delta:.3g}): {sol.message}")

    result = (sol.x, sol.y)
    _carreau_bvp_cache[cache_key] = result
    return result

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
