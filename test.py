import numpy as np

import matplotlib.pyplot as plt

import control as ctrl

from Constraints import K, Kc,A,B,C


def plot_SRG_simulation(time_interval, xx, target_state, kappas):
    """Plots the results of the SRG simulation

    Args:
        time_interval (Vector): A vector of time steps
        xx (Matrix): The stacked matrix containing states at each time step from Euler simulation
        target_state (Vector): The target states for X, Y, and Z
        kappas (Vector): The values of kappa over time
    """

    # Plotting results
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"Reference Governor Flight. \n Target state X: {target_state[1]}, Y: {target_state[3]}, and Z: {target_state[5]}")

    # Change in X over time
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(time_interval, xx[1, :], label='X Change')
    ax1.set_title('Change in X')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X Position')

    # Change in Y over time
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(time_interval, xx[3, :], label='Y Change', color='orange')
    ax2.set_title('Change in Y')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Y Position')

    # Change in Z over time
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(time_interval, xx[5, :], label='Z Change', color='green')
    ax3.set_title('Change in Z')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Z Position')

    # 3D plot of X, Y, Z states
    ax4 = fig.add_subplot(3, 2, 4, projection='3d')
    ax4.plot(xx[1, :], xx[3, :], xx[5, :],
             label='3D Trajectory', color='purple')
    ax4.set_title('3D State Trajectory')
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    ax4.set_zlabel('Z Position')

    # Kappas over time
    ax5 = fig.add_subplot(3, 1, 3)
    ax5.plot(time_interval[::10], kappas[:len(
        time_interval[::10])], label='Kappas', color='red')
    ax5.set_title('Kappas over Time')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Kappa Value')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
    plt.show()


def plot_SRG_controls(time_interval, controls, target_state):
    """Plots the control variables with constraints for the SRG simulation.

    Args:
        time_interval (Vector): A vector of time steps.
        controls (Matrix): A 4xN matrix where each row represents a control variable over time.
        target_state (Vector): The target state
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"SRG Control Variables with Constraints and Target X = {target_state[1]}, Y = {target_state[3]}, Z = {target_state[5]}")

    # Plot for controls[0]
    axs[0, 0].plot(time_interval, controls[0, :], label='Control 1')
    axs[0, 0].axhline(y=6, color='red', linestyle='--',
                      label='Constraint at 6')
    axs[0, 0].set_title('Control Variable 1')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Control 1')
    axs[0, 0].legend()

    # Plot for controls[1]
    axs[0, 1].plot(time_interval, controls[1, :], label='Control 2')
    axs[0, 1].axhline(y=0.005, color='red', linestyle='--',
                      label='Constraint at 0.005')
    axs[0, 1].set_title('Control Variable 2')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Control 2')
    axs[0, 1].legend()

    # Plot for controls[2]
    axs[1, 0].plot(time_interval, controls[2, :], label='Control 3')
    axs[1, 0].axhline(y=0.005, color='red', linestyle='--',
                      label='Constraint at 0.005')
    axs[1, 0].set_title('Control Variable 3')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Control 3')
    axs[1, 0].legend()

    # Plot for controls[3]
    axs[1, 1].plot(time_interval, controls[3, :], label='Control 4')
    axs[1, 1].axhline(y=0.005, color='red', linestyle='--',
                      label='Constraint at 0.005')
    axs[1, 1].set_title('Control Variable 4')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Control 4')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
    
def construct_h(s, epsilon, ell_star):
    """Construct the contraint matrix h

    Args:
        s (vector): The constraint vector
        epsilon (float): A small positive number
        ell_star (int): number of iterations (timesteps)

    Returns:
        matrix: The constraint matrix h
    """

    h = [s] * ell_star

    # Last element is s - epsilon (epsilon is small positive number)
    h.append(s - epsilon)

    return np.array(h)

def SRG_Simulation_Linear(desired_state, time_steps=0.0001,
                          initial_state=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
    """The state-reference governor simulation with linear dynamics

    Args:
        desired_state (vector): The target state
        time_steps (float, optional): The time steps. Defaults to 0.0001.
        initial_state (vector, optional): The initial state. Defaults to np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).

    Returns:
        Matrix: The state-matrix of the Euler simulation
    """
    x0 = initial_state

    # Transforming the desired (target) point into a 4x1 vector
    desired_coord = np.array([desired_state[i]
                             for i in [1, 3, 5, 11]]).reshape(-1, 1)
    print(desired_coord)

    # Initial feasible control vk (a 4x1 vector)
    # (the first valid point along the line from A to B), this serves as the starting point
    vk = 0.01 * desired_coord

    # ell_star is the number of iterations to perform when forming the contraint matrices Hx, Hv, and h
    ell_star = 100000
    # make ell_star to be 100 when doing figure 8
    # ell_star = 100

    # ime interval for the continuous-time system
    time_interval = np.arange(0, 10 + time_steps, time_steps)

    # Number of time steps for Euler’s method loop
    N = len(time_interval)

    # S is a constraint matrix that uses the feedback matrices K and Kc. S @ x needs to be less than the constraints
    S = np.block([[-K, Kc],
                  [K, -Kc]])

    # Defining the blocks for integral control, we need to use discrete versions of A_cl and B_cl
    # so we obtain Ad and Bd to use in governor
    Ac = np.zeros((4, 4))
    Bc = np.eye(4)

    Acl = np.block([
        [A - B @ K, B @ Kc],
        [-Bc @ C, Ac]
    ])

    Bcl = np.vstack([np.zeros((12, 4)), Bc])
    Ccl = np.hstack((C, np.zeros((C.shape[0], C.shape[0]))))
    Dcl = np.zeros((C.shape[0], B.shape[1]))

    sys = ctrl.ss(Acl, Bcl, Ccl, Dcl)
    sysd = ctrl.c2d(sys, time_steps)

    # The final discrete matrices to use in the closed-loop system
    Ad = sysd.A
    Bd = sysd.B

    # Constructing contraint matrices and constraint vector s
    Hx = construct_Hx(S, Ad, ell_star)
    Hv = construct_Hv(S, Ad, Bd, ell_star)
    print(Hv)
    s = np.array([6, 0.005, 0.005, 0.005, 6, 0.005, 0.005, 0.005]).T
    epsilon = 0.005
    h = construct_h(s, epsilon, ell_star)

    # Initialize x array (evolving state over time)
    xx = np.zeros((16, N))
    xx[:, 0] = x0.flatten()


    # The control function for Euler simulation
    def qds_dt(x, uc, Acl, Bcl):
        """Defines the change in state for the first 12 (non-integral)
        Args:
            x (vector): The current state
            u (vector): The current error
            Acl (matrix): A control matrix
            Bcl (matrix): B control matrix

        Returns:
            vector: The change in the state
        """

        # Integral control (linear version)
        
        return (Acl @ x + (Bcl @ uc.reshape(4, 1)).reshape(1, 16)).reshape(16)

    # Main simulation loop for SRG Euler simulation
    # Sampling time for reference governor (ts1 > time_steps)
    ts1 = 0.001

    controls = np.zeros((4, N))
    kappas = []

    for i in range(1, N):

        t = (i - 1) * time_steps

        if (t % ts1) < time_steps and i != 1:

            # Getting kappa_t solution from reference governor
            # We select the minimum feasible kappa-star as the solution
            kappa = min(rg(Hx, Hv, h, desired_coord, vk, xx[:, i - 1], i-1), 1)
            kappas.append(kappa)

            # Updating vk
            vk = vk + kappa * (desired_coord - vk)

        # Integral control
        u = -K @ xx[:12, i - 1] + Kc @ xx[12:16, i - 1]

        controls[:, i] = u.reshape(1, 4)[0]

        xx[12:, i] = xx[12:, i-1] + \
            (vk.reshape(1, 4)[0] - xx[[1, 3, 5, 11], i-1]) * time_steps

        xx[:12, i] = xx[:12, i-1] + \
            qds_dt(xx[:, i-1], u, Acl, Bcl)[:12] * time_steps

    return xx, controls, time_interval, np.array(kappas)

    # Reference governor computation  
def rg(Hx, Hv, h, desired_coord, vk, state, j):
    """The scalar reference governor returns a scalar values (one) representing the maximum feasible step
        toward the desired coordinates.

    Args:
        Hx (matrix): A constraint matrix
        Hv (matrix): A constraint matrix
        h (matrix): A constraint matrix
        desired_coord (vector): The desired coordinates for the quadrotor
        vk (vector): The control
        state (vector): The current state
        j (int): An index for the simulation loop

    Returns:
        float: A kappa value
    """
    kappa_list = []
    j = min(1, h.shape[0] - 1)
   
    #print(h[j].shape[0])
    # Computing K*_j for each constraint inequality
    for i in range(h[j].shape[0]):

        Bj = h[j][i] - (Hx[j] @ state) - (Hv[j] @ vk)

        Aj = Hv[j].T @ (desired_coord - vk)

        # add check here for bj ? Like he said in notes..?
        # Add check for Bj < 0
        # if Bj < 0:
        #     kappa = 0
        #     kappa_list.append(kappa)
        
        # If Aj <= 0, we set kappa-star to 1
        if Aj <= 0:
            kappa = 1
            kappa_list.append(kappa)

        else:

            kappa = Bj / Aj

            # If kappa is infeasible
            if kappa < 0 or kappa > 1:
                kappa = 0

            kappa_list.append(kappa)

    # Min kappa-star out of the 8 inequalities is optimal solution
    return min(kappa_list)
def calculate_control(curstate_integral):
    """calculate control
    Args:
        curstate_integral (array): Current state 
    Returns:
        array: 1x4 control vector
    """
    control = np.block([-K, Kc]) @ curstate_integral
    return control

    # This function constructs the constraint matrix Hx (which places constraints on states)
def construct_Hx(S, Ad, ell_star):
    """Construct the Hx contraint matrix Hx
    Args:
        S (matrix): A constraint block matrix of K and Kc
        Ad (matrix): Discrete A control matrix
        ell_star (int): number of iterations (timesteps)
    Returns:
        matrix: The constraint matrix Hx
    """

    Hx = []

    # First element is Sx
    Hx.append(S)

    # For every time step, construct new constraints on the state
    for ell in range(ell_star + 1):
        Ax = np.linalg.matrix_power(Ad, ell)
        Hx.append(S @ Ax)

    return np.vstack(Hx)
def construct_Hv(S, Ad, Bd, ell_star):
    """Construct the Hv constraint matrix
    Args:
        S (matrix): A constraint block matrix of K and Kc
        Ad (matrix): Discrete A control matrix
        x (vector): Future update of state using "predict_future_state()" function
        ell_star (int): number of iterations (timesteps)
        Bd (matrix): Discrete B control matrix

    Returns:
        matrix: The constraint matrix Hv
    """
    # First element is 0
    Hv = [np.zeros((S.shape[0], Bd.shape[1]))]
    Hv.append(S @ Bd)

    # For every time step, construct new constraints on the control
    for ell in range(1, ell_star + 1):

        # Calculate A_d^ℓ
        Ad_ell = np.linalg.matrix_power(Ad, ell)

        I = np.eye(Ad.shape[0])
        Ad_inv_term = np.linalg.inv(I - Ad)

        I_minus_Ad_ell = I - Ad_ell

        # Compute the entire expression
        result = S @ Ad_inv_term @ I_minus_Ad_ell @ Bd

        Hv.append(result)

    return np.vstack(Hv) 
if __name__ == '__main__':
    print("Main started")

    target_state = [
        0, 10,   # velocity and position on x
        0, 10,    # velocity and position on y
        0, 10,    # velocity and position on z
        0, 0,     # angular velocity and position thi
        0, 0,     # angular velocity and position thetha
        0, 0]     # angular velocity and position psi ]

    target_state_16 = [
        0, 10,   # velocity and position on x
        0, 10,    # velocity and position on y
        0, 10,    # velocity and position on z
        0, 0,     # angular velocity and position thi
        0, 0,     # angular velocity and position thetha
        0, 0,     # angular velocity and position psi ]
        0, 0,     # Integral states are 0s
        0, 0]

    # Run simulation for EULERS METHOD: This should work like it is Part1 in project writeup
    # results, control, time_interval = simulate_nonlinear_integral_with_euler(target_state=target_state_16)
    # results, control, time_interval = simulate_linear_integral_with_euler(target_state=target_state_16)
    # plot_SRG_simulation(time_interval, results, target_state_16, kappas=[0]*10001)

    
# ----------------------------------------------------------------

    # xx, controls, time_interval, kappas= SRG_Simulation_Linear(desired_state=target_state_16)
    # xx, controls, time_interval, kappas = SRG_Simulation_Nonlinear(desired_state=target_state_16)
    
    # plot_SRG_simulation(time_interval, xx,
    #                     target_state=target_state_16, kappas=kappas)
    
    # plot_SRG_controls(time_interval, controls, target_state)
# ------------------------------------------------------------------



    waypoints1 = [
        [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 3, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]


# stopped here, maybe probelm in how new desired state is made???
# double chekc if SRG_Simulation_NOnlinear before adding waypoints still works


    #xx, controls, time_interval, kappas = SRG_Simulation_Linear(desired_state=target_state_16, ell_star_figure8=True, use_multiple_waypoints=True, waypoints=waypoints1)

    #plot_SRG_simulation(time_interval, xx,target_state=target_state_16, kappas=kappas)
    
# def SRG_Simulation_Nonlinear(desired_state, time_steps=0.0001,
#                              initial_state=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ell_star_figure8=False, use_multiple_waypoints=False, waypoints=None):



    # Chris Function: Takes forever????
    # simulate_figure_8_srg(At=9, Bt=33, omega=.5, z0=12)
    
    # simulate_figure8_srg_max()


# I think old stuff??    # simulate_figure_8()
    # simulate_quadrotor_nonlinear_controller(target_state=target_state)
    # print(f'Max force before bound: {np.max(force_before_bound)}')
    # print(f'Max force after bound: {np.max(force_after_bound)}')

    # clear_bound_values()
    # simulate_quadrotor_nonlinear_controller(
    # target_state=target_state, bounds=(6, 0))

    # simulate_quadrotor_linear_integral_controller(target_state=target_state_16)

    # Have not tested, or verified this simulate_figure_8 funtion
    # simulate_figure_8(At=9, Bt=33, omega=.5, z0=12)



        # # clear_bound_values()
    # simulate_quadrotor_nonlinear_controller(target_state)
    # print(f'Max force before bound: {np.max(force_before_bound)}')
    # simulate_quadrotor_linear_controller(target_state, bounds=(0.4, 0))
    # print(f'Max force after bound: {np.max(force_after_bound)}')

    # clear_bound_values()
    

    # clear_bound_values()
    xx, controls, time_interval, kappas= SRG_Simulation_Linear(desired_state=target_state_16)
    print("xx:",xx)
    print(kappas)
    print("controls,", controls)
    plot_SRG_simulation(time_interval, xx,target_state=target_state_16, kappas=kappas)
    plot_SRG_controls(time_interval, controls, target_state=target_state_16)
    
    
