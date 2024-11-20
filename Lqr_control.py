import numpy as np
from scipy.integrate import solve_ivp
import scipy.io



# Load K and Kc matrices
K = scipy.io.loadmat('K.mat')['K']
Kc = scipy.io.loadmat('Kc.mat')['Kc']

# Drone parameters
M = 0.6  # mass of the quadrotor (Kg)
L = 0.2159  # arm length (m)
g = 9.81  # acceleration due to gravity (m/s^2)
m = 0.410  # mass of the central sphere (Kg)
R = 0.0503513  # radius of the sphere (m)
m_prop = 0.00311  # mass of the propeller (Kg)
m_m = 0.036 + m_prop  # mass of the motor + propeller (Kg)

# Moments of inertia
Jx = (2 * m * R) / 5 + 2 * L**2 * m_m
Jy = (2 * m * R) / 5 + 2 * L**2 * m_m
Jz = (2 * m * R) / 5 + 4 * L**2 * m_m

# System matrices
A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
])

B = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-1/M, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1/Jx, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1/Jy, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1/Jz],
    [0, 0, 0, 0]
])

C = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
target_state = [0, 30, 0, 30, 0, 30, 0, 0, 0, 0, 0, 0]
time_span = [0, 5]

# Function for linear dynamics
def intergral_linear(time, current_state, Ac, Bc, target_state, K): # use Ac and Bc instead of A and B
    control =  (current_state[[1,3,5,11]] - target_state[[1,3,5,11]])
    dx = Ac @ current_state + Bc @ control
    return dx

# LQR control function to compute A_cl and B_cl and solve for LQR dynamics
def integral_Control(A, B, C, Kc, K):
    n = A.shape[0]
    p = B.shape[1]
    q = C.shape[0]
    
    # Augmented initial and target states
    aug_initial_state = np.concatenate([initial_state, np.zeros(q)])
    aug_target_state = np.concatenate([target_state, np.zeros(q)])
    
    # Augmented system matrices
    Ac = np.zeros((q, q))
    Bc = np.eye(q)
    
    Acl = np.block([[A - B @ K, B @ Kc],
                    [-Bc @ C, Ac]])
    Bcl = np.vstack([np.zeros((n, p)), Bc])
    
    # Solve LQR dynamics using solve_ivp
    sol_lqr = solve_ivp(
        intergral_linear, time_span, aug_initial_state,
        args=(Acl, Bcl, aug_target_state, K),
        dense_output=True
    )
    print(sol_lqr.y)
    # Extract positions for plotting
    x_lqr, y_lqr, z_lqr = sol_lqr.y[1], sol_lqr.y[3], sol_lqr.y[5]  # LQR Linear positions
    t = sol_lqr.t  # Time vector
    
    # Display plot
    display_plot(t, x_lqr, y_lqr, z_lqr)

# Main function
def main():
    integral_Control(A, B, C, Kc, K)

# Run the main function
if __name__ == "__main__":
    main()
