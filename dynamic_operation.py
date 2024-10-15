import numpy as np
from scipy.integrate import solve_ivp

# Import the constants from Constraints
from Constraints import mq, g, Jx, Jy, Jz, K, A, B,K

def non_linear(time,current_state, target_state,K):
    '''
    Nonlinear dynamics of the quadrotor.
    Returns the derivative of the state.
    '''

    # the current state
    u, px, v, py, w, pz, p, phi, q, theta, r, psi = current_state

    # Control law (feedback control based on error)
    control = -K @ (current_state - target_state)

    # Adding gravity effect to the control for z-axiz thrust
    control[0] += mq * g
    
    # Thrust force
    F = control[0]

    # Linear velocity dynamics (u_dot, v_dot, w_dot)
    u_dot = r*v - q*w - g*np.sin(theta)
    v_dot = p*w - r*u + g*np.cos(theta)*np.sin(phi)
    w_dot = q*u - p*v + g*np.cos(theta)*np.cos(phi) - F/mq

    # Angular velocity dynamics (p_dot, q_dot, r_dot)
    p_dot = (Jy - Jz) / Jx * q * r + control[1] / Jx  # control[1] corresponds to torque τ_φ
    q_dot = (Jz - Jx) / Jy * p * r + control[2] / Jy  # control[2] corresponds to torque τ_θ
    r_dot = (Jx - Jy) / Jz * p * q + control[3] / Jz  # control[3] corresponds to torque τ_ψ

    #return the derivatives
    dx = np.array([u_dot,u,v_dot,v,w_dot,w, p_dot,p, q_dot,q,r_dot,r])

    return dx


def linear(time,current_state, A,B, target_state,K):
    """
    Computes the linear dynamics of the drone system.
    Returns dx
    """
    # Control law (feedback control based on state error)
    control = -K @ (current_state - target_state)
    # Linear state update: dx = Ax + B * control
    dx = A @ current_state + B @ control
    
    return dx

def solve_quadrotor_dynamics(initial_state, target_state, time_span, time_eval):
    # Solve nonlinear dynamics
    sol_non_linear = solve_ivp(non_linear, time_span, initial_state, args=(target_state, K), t_eval=time_eval)
    
    # Solve linear dynamics
    sol_linear = solve_ivp(linear, time_span, initial_state, args=(A, B, target_state, K), t_eval=time_eval)
    
    return sol_non_linear, sol_linear
    
