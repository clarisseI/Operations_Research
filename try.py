import numpy as np
from Constraints import *  # Import system-specific constants
from Figure_8_Trajectory import generate_waypoints_figure_8  # For trajectory generation

def Hx_matrix(S, Ad, ell_star):
    """ Compute Hx matrix based on constraints and system dynamics. """
    hrows = []
    for l in range(ell_star + 1):
        Ad_Power = np.linalg.matrix_power(Ad, l)
        hrow = S @ Ad_Power
        hrows.append(hrow)
    return np.vstack(hrows)

def Hv_matrix(S, Ad, Bd, ell_star):
    """ Compute Hv matrix for control constraints over future steps. """
    hvrows = []
    I = np.eye(Ad.shape[0])
    Ad_inv = np.linalg.inv(I - Ad)
    for l in range(ell_star + 1):
        Ad_Power = np.linalg.matrix_power(Ad, l)
        Ad_Power_Diff = I - Ad_Power
        hvrow = S @ Ad_inv @ Ad_Power_Diff @ Bd
        hvrows.append(hvrow)
    return np.vstack(hvrows)

def H_matrix(s, epsilon, ell_star):
    """ Generate the matrix H with epsilon adjustment on the last row. """
    h = np.tile(s, (ell_star - 1, 1))
    h = np.vstack([h, s - epsilon])
    return h

def compute_kappa(Hx, Hv, h, ref, vk, x0):
    """ Solve the RG problem to find kappa. """
    kappa_total = []
    for j in range(Hx.shape[0]):
        Betta = h[j] - (Hx[j] @ x0) - (Hv[j] @ vk[:1])
        Alpha = Hv[j].T @ (ref - vk)
        kappa = Betta / Alpha if Alpha > 0 else 1
        kappa = max(0, min(1, kappa))  # Constrain kappa between 0 and 1
        kappa_total.append(kappa)
    return min(kappa_total)

def qds_dt_linear(current_state, control, Acl, Bcl):
    """ Linear dynamics function. """
    return Acl @ current_state + Bcl @ control

def SRG_linear(S, Ad, Bd, K, Kc, s, epsilon, ell_star, ts, ts1, ref, Acl, Bcl):
    """ Main function to compute SRG with the given dynamics. """
    # Initializations
    x0 = np.zeros(12)  # Initial state
    xc = np.zeros(4)   # Integral control state
    vk = 0.01 * np.array(ref)  # Initial feasible control
    tt = np.arange(0, 10 + ts, ts)  # Time interval
    N = len(tt)
    xx = np.zeros((16, N))
    xx[:, 0] = np.hstack([x0, xc])  # Initial states
    u = np.zeros((4, N))  # Control input
    kappa_total = []

    # Compute constraint matrices
    Hx = Hx_matrix(S, Ad, ell_star)
    Hv = Hv_matrix(S, Ad, Bd, ell_star)
    print(Hv)
    H = H_matrix(s, epsilon, ell_star)

    # Iterate over time steps
    for i in range(1, N):
        t = (i - 1) * ts
        if t % ts1 < ts:
            kappa = compute_kappa(Hx, Hv, H, ref, vk, xx[:12, i - 1])
            kappa_total.append(kappa)
            vk += kappa * (ref - vk)

        u[:, i] = -K @ xx[:12, i - 1] + Kc @ xx[12:, i - 1]
        xx[12:, i] = xx[12:, i - 1] + (vk - xx[[1, 3, 5, 11], i - 1]) * ts
        xx[:12, i] = xx[:12, i - 1] + qds_dt_linear(xx[:12, i - 1], u[:, i], Acl, Bcl) * ts

    return xx, xc, tt, kappa_total

def main():
    # Define constants and parameters
    ts = 0.01  # Sampling time
    ts1 = 0.1  # Reference update interval
    ell_star = 1000  # Prediction horizon
    epsilon = 0.001  # Constraint adjustment parameter
    ref = [3, 3, 3, 0]  # Desired reference point

    # Import or define matrices S, Ad, Bd, K, Kc, Acl, Bcl, s
    # Example placeholders:
    S = np.eye(12)
    Ad = np.eye(12)
    Bd = np.ones((12, 4))
    K = np.random.rand(4, 12)
    Kc = np.random.rand(4, 4)
    s = np.ones(12)
    Acl = np.eye(12)
    Bcl = np.ones((12, 4))

    # Run SRG Linear
    xx, xc, tt, kappa_total = SRG_linear(S, Ad, Bd, K, Kc, s, epsilon, ell_star, ts, ts1, ref, Acl, Bcl)

    # Print results
    print("State Trajectories (xx):", xx)
    print("Control States (xc):", xc)
    print("Time Steps (tt):", tt)
    print("Kappa Values:", kappa_total)

if __name__ == "__main__":
    main()
