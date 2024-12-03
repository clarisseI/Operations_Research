'''
High distance= breaks
create small steps
find maximum feasible steps and as less stops as possible
The SRG algorithm adjusts the reference trajectory  by computing  κt, the maximum feasible step from the current reference 
The system iterates over time to update the reference trajectory and ensure constraint satisfaction at each step.
The SRG is formulated as a linear programming (LP) problem with constraints based on the system dynamics.

'''
from Constraints import * 
from Figure_8_Trajectory import generate_waypoints_figure_8

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

    
def Hx_Matrix(S, Ad, ell_star):
       
    hrows = []

    # Compute each term in the series S * A^l (for each l from 0 to l_star)
    for l in range(0, ell_star + 1):
        # Compute the matrix A^l
        Ad_Power = np.linalg.matrix_power(Ad, l)
            
        # Compute the hrow for the current step
        hrow = S @ Ad_Power
            
        # Append the computed hrow to the list
        hrows.append(hrow)
        
    # Stack all hrows vertically to form the full Hx matrix
    Hx = np.vstack(hrows)
        
    return Hx
       

def Hv_Matrix(S, Ad, Bd, ell_star):
    
    # First element is 0
    Hv = [np.zeros((S.shape[0], Bd.shape[1]))]
    

    # For every time step, construct new constraints on the control
    for ell in range(0, ell_star + 1):

        # Calculate A_d^ℓ
        Ad_ell = np.linalg.matrix_power(Ad, ell)

        I = np.eye(Ad.shape[0])
        Ad_inv_term = np.linalg.inv(I - Ad)

        I_minus_Ad_ell = I - Ad_ell

        # Compute the entire expression
        result = S @ Ad_inv_term @ I_minus_Ad_ell @ Bd

        Hv.append(result)

    return np.vstack(Hv) 
    
    
def H_Matrix( s,epsilon, ell_star):
     # Create the matrix where all rows are initially s
        h = np.tile(s, (ell_star - 1, 1))  # Repeat s for (l_star - 1) times

        # Subtract epsilon for the last row
        h = np.vstack([h, s - epsilon])

        return h
       
       
        
def rg( Hx, Hv, H, ref, vk, x0,current_step):
    """
    Compute the minimum kappa for the reference governor.
    """
    """
    Reference governor function that computes the minimum kappa without explicitly using 'j'.
    """
    kappa_total = []
    
    # Dynamically determine j based on the current step or state
    j = min(current_step, H.shape[0] - 1)

    for i in range(H[j].shape[0]):
        Betta = H[j][i] - (Hx[j] @ x0) - (Hv[j] @ vk)
        Alpha = Hv[j].T @ (ref - vk)

        # If Aj <= 0, set kappa to 1
        if Alpha <= 0:
            kappa = 1
            kappa_total.append(kappa)
        else:
            kappa = Betta / Alpha
            # Ensure kappa is feasible
            if kappa < 0 or kappa > 1:
                kappa = 0

            kappa_total.append(kappa)

    # Return the minimum kappa
    return min(kappa_total)   
def qds_non_linear(current_state,control):
    # Dynamics for nonlinear model (same as before)
    u, px, v, py, w, pz, p, phi, q, theta, r, psi = current_state

   
    control[0] += M * g  # gravity effect

    F = control[0]

    # Velocity dynamics
    u_dot = r * v - q * w - g * np.sin(theta)
    v_dot = p * w - r * u + g * np.cos(theta) * np.sin(phi)
    w_dot = q * u - p * v + g * np.cos(theta) * np.cos(phi) - F / M

    # Angular velocity dynamics
    p_dot = (Jy - Jz) / Jx * q * r + control[1] / Jx
    q_dot = (Jz - Jx) / Jy * p * r + control[2] / Jy
    r_dot = (Jx - Jy) / Jz * p * q + control[3] / Jz

    # Derivatives
    dx = np.array([u_dot, u, v_dot, v, w_dot, w, p_dot, p, q_dot, q, r_dot, r])
   
    return dx
     
def qds_linear(current_state, control):
        
    return A @ current_state + B @ control
        
def SRG_linear(model="linear"):
    
    ts = 0.001 # Sampling time
    ts1= 0.01
    x0 = np.zeros((16, 1)) # Initial state
  
    
    ref = np.array([3, 3, 3, 0])
   
    vk = 0.01 * ref # Initial feasible v0
        
    ell_star = 1000
        
    tt = np.arange(0, 10+ts, ts) #[0:ts:10] #Time interval for the continuous-time system
        
    N = tt.size #Number of time steps for Euler’s method
    
    cu_total = np.zeros((4, N))   # Control state
        
    Hx = Hx_Matrix(S, Ad, ell_star)
    
    Hv = Hv_Matrix(S, Ad, Bd, ell_star)
    
    epsilon=0.001
    
    H= H_Matrix (s,epsilon, ell_star)
    
    xx= np.zeros((16, N))  

    xx[:, 0] = x0.flatten()
 
    kappa_total=[]
    
    cu= np.zeros((4, N))
    
        
    for i in range(2,N):
        t= (i-1)* ts

        # update the reference governor
        if t % ts1 < ts :
            kappa = min(rg(Hx, Hv, H, ref, vk, xx[:, i - 1], i - 1), 1)  # Pass i-1 as current_step
            kappa_total.append(kappa)
                
            vk= vk + kappa *(ref-vk)
            
        #compute integral control
        cu[:, i]= -K @ xx[:12, i-1] + Kc @ xx[12:16, i-1]
       
        
        ## for non-linear
        #u, xc= qds_non_linear(xx[:, i-1], ref)

        xx[12:, i]= xx[12:, i-1] + \
                          (vk.reshape(1, 4)[0]- xx[[1,3,5,11], i-1]) *ts
        
        if model == 'linear':
            xx[:12, i]= xx[:12, i-1]+ \
                                    qds_linear(xx[:12,i-1],cu[:, i])*ts
            
        elif model == 'nonlinear':
            #state_change,u= qds_non_linear(xx[:,i-1],ref)
           
            xx[:12, i]= xx[:12, i-1]+ \
                            qds_non_linear(xx[:12,i-1],cu[:, i])*ts

       # store current state
       
        # Store cu values into cu_total
        cu_total[:, i] = cu[:, i]
        
    return xx,cu_total, tt,np.array(kappa_total)

def plot_xyz_side_by_side(time_eval, x_total_linear, x_total_nonlinear):
    """Plots x, y, and z positions for linear and nonlinear models side by side."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Reference Governor: Linear vs Nonlinear Dynamics", fontsize=16)

    labels = ["x-position", "y-position", "z-position"]
    colors = ["blue", "green", "red"]
    linear_titles = ["Linear Dynamics"] * 3
    nonlinear_titles = ["Nonlinear Dynamics"] * 3

    # Indices for positions in the state vector for both linear and nonlinear models
    linear_position_indices = [1, 3, 5]  # x, y, z for linear model
    nonlinear_position_indices = [1, 3, 5]  # x, y, z for nonlinear model

    for i, (label, color, linear_title, nonlinear_title) in enumerate(zip(labels, colors, linear_titles, nonlinear_titles)):
        # Linear plots
        axes[i, 0].plot(time_eval, x_total_linear[linear_position_indices[i], :], label=f"Linear {label}", color=color)
        axes[i, 0].set_title(linear_title, fontsize=12)
        axes[i, 0].set_ylabel(f"{label} (m)", fontsize=10)
        axes[i, 0].set_xlabel("Time (s)", fontsize=10)
        axes[i, 0].grid()
        axes[i, 0].legend()

        # Nonlinear plots
        axes[i, 1].plot(time_eval, x_total_nonlinear[nonlinear_position_indices[i], :], label=f"Nonlinear {label}", color=color)
        axes[i, 1].set_title(nonlinear_title, fontsize=12)
        axes[i, 1].set_ylabel(f"{label} (m)", fontsize=10)
        axes[i, 1].set_xlabel("Time (s)", fontsize=10)
        axes[i, 1].grid()
        axes[i, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def compare_xyz_positions(time_eval, x_total_linear, x_total_nonlinear):
    
    # Extract positions for X, Y, Z for both linear and nonlinear models
    x_linear = x_total_linear[1, :]
    y_linear = x_total_linear[3, :]
    z_linear = x_total_linear[5, :]
    
    x_non_linear = x_total_nonlinear[1, :]
    y_non_linear = x_total_nonlinear[3, :]
    z_non_linear = x_total_nonlinear[5, :]

    # Calculate absolute differences
    diff_x = np.abs(x_linear - x_non_linear)
    diff_y = np.abs(y_linear - y_non_linear)
    diff_z = np.abs(z_linear - z_non_linear)
    
    # Calculate average differences
    avg_diff_x = np.mean(diff_x)
    avg_diff_y = np.mean(diff_y)
    avg_diff_z = np.mean(diff_z)

    # Print average differences with formatting
    print(f"\n{'Average Absolute Differences for Scalar Reference Governor':^40}")
    print("-" * 40)
    print(f"Average absolute difference for X: {avg_diff_x:.4f}")
    print(f"Average absolute difference for Y: {avg_diff_y:.4f}")
    print(f"Average absolute difference for Z: {avg_diff_z:.4f}")
    print("-" * 40)

    # Sample 10 values for better insight
    # Ensure the number of samples does not exceed the length of time_eval
    sample_size = min(10, len(time_eval))
    sample_indices = np.linspace(0, len(time_eval) - 1, sample_size, dtype=int)
    
    # Prepare data for tabulate
    table_data = []
    for i in sample_indices:
        table_data.append([
            f"{time_eval[i]:.2f}",
            f"{x_linear[i]:.4f}",
            f"{y_linear[i]:.4f}",
            f"{z_linear[i]:.4f}",
            f"{x_non_linear[i]:.4f}",
            f"{y_non_linear[i]:.4f}",
            f"{z_non_linear[i]:.4f}"
        ])

    # Create the table headers
    headers = ["Time (s)", "X_Linear", "Y_Linear", "Z_Linear", "X_Nonlinear", "Y_Nonlinear", "Z_Nonlinear"]

    # Display the table with nice formatting
    print("\n10 Sampled Positions for Linear and Nonlinear Trajectories for Scalar Reference Governor:")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", numalign="center", stralign="center"))



def plot_controls(xc, tt, title="Control Inputs with Constraints"):
    """
    Plots control inputs with constraints for linear or nonlinear models.

    Parameters:
    - xc: Control input array (shape: [4, N])
    - tt: Time array (shape: [N])
    - title: Title for the entire plot (default: "Control Inputs with Constraints")
    """
    # Create a 2x2 grid for the subplots to optimize space usage
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Line styles for the plots
    line_styles = ['-', '--', '-.', ':']
    colors = ['c', 'm', 'y', 'orange']
    labels = ['Force', 'Tau-Phi', 'Tau-Theta', 'Tau-Psi']
    constraints = [6, 0.005, 0.005, 0.005]

    # Loop through each control input and plot it
    for j in range(4):
        # Plot the control input
        axs[j//2, j%2].plot(tt, xc[j, :], label=labels[j], color=colors[j], linestyle=line_styles[j%4], linewidth=2)

        # Add the constraint line
        axs[j//2, j%2].axhline(y=constraints[j], color='red', linestyle='--', linewidth=1.5)

        # Set labels and titles
        axs[j//2, j%2].set_xlabel('Time (s)', fontsize=12)
        axs[j//2, j%2].set_ylabel('Control Input', fontsize=12)
        axs[j//2, j%2].set_title(labels[j], fontsize=10, fontweight='bold')
        axs[j//2, j%2].legend(fontsize=10)
        axs[j//2, j%2].grid(True)

        # Adjust y-limits to make sure the constraint line is visible
        if j == 0:  # For Force, set limit based on constraint
            axs[j//2, j%2].set_ylim(min(np.min(xc[0, :]) - 0.1, 0), 6 + 0.1)
        else:
            axs[j//2, j%2].set_ylim(np.min(xc[j, :]) - 0.001, np.max(xc[j, :]) + 0.001)

    # Add the dynamic title for the entire figure
    fig.suptitle(title, fontsize=18, fontweight='bold')

    # Optimize layout to avoid overlap, increase the space for title
    plt.tight_layout(pad=3.0, h_pad=2.0, rect=[0, 0, 1, 0.96])

    plt.show()
def plot_control_inputs_side_by_side_try(time_eval, cu_total_linear, cu_total_nonlinear):
    """Plots the control inputs F, tau-phi, tau-theta, and tau-psi for linear and nonlinear models side by side."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle("Control Inputs (Linear vs Nonlinear) for F, Tau-Phi, Tau-Theta, Tau-Psi", fontsize=12)

    labels = ["F ", "Tau-Phi ", "Tau-Theta ", "Tau-Psi "]
    control_indices = [0, 1, 2, 3]  # F, Tau-Phi, Tau-Theta, Tau-Psi
    constraints = [6, 0.005, 0.005, 0.005]  # Constraints for each control input

    for i, (label, idx) in enumerate(zip(labels, control_indices)):
        # Plot linear control inputs
        axes[i, 0].plot(time_eval, cu_total_linear[idx, :], label=f"Linear {label}", color="blue")
        axes[i, 0].set_title(f"Linear {label}", fontsize=10)
        axes[i, 0].set_ylabel(f"{label} ", fontsize=8)
        axes[i, 0].set_xlabel("Time (s)", fontsize=8)
        axes[i, 0].grid()
        axes[i, 0].legend()
        # Add constraint lines for linear model
        axes[i, 0].axhline(y=constraints[idx], color='red', linestyle='--', linewidth=1.5)
        axes[i, 0].axhline(y=-constraints[idx], color='red', linestyle='--', linewidth=1.5)
        # Plot nonlinear control inputs
        axes[i, 1].plot(time_eval, cu_total_nonlinear[idx,:], label=f"Nonlinear {label}", color="red")
        axes[i, 1].set_title(f"Nonlinear {label}", fontsize=10)
        axes[i, 1].set_ylabel(f"{label} ", fontsize=8)
        axes[i, 1].set_xlabel("Time (s)", fontsize=8)
        axes[i, 1].grid()
        axes[i, 1].legend()
        # Add constraint lines for nonlinear model
        axes[i, 1].axhline(y=constraints[idx], color='green', linestyle=':', linewidth=1.5)
        axes[i, 1].axhline(y=-constraints[idx], color='green', linestyle=':', linewidth=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
def check_constraints(xc_total_linear, xc_total_nonlinear):
    """Check if the control inputs F, Tau-Phi, Tau-Theta, and Tau-Psi violate the constraints."""
    labels = ["F", "Tau-Phi", "Tau-Theta", "Tau-Psi"]
    control_indices = [0, 1, 2, 3]  # F, Tau-Phi, Tau-Theta, Tau-Psi
    #constraints_min = [-6, -0.005, -0.005, -0.005]
    constraints_max = [6, 0.005, 0.005, 0.005]

    table_data = []

    for label, idx in zip(labels, control_indices):
        # Find the indices where the control inputs violate the constraints
        # Check linear control violations using absolute value
        linear_violations = np.where(np.abs(xc_total_linear[idx,:]) > constraints_max[idx])[0]
        nonlinear_violations = np.where(np.abs(xc_total_nonlinear[idx,: ] >= constraints_max[idx]))[0]

        # Prepare the violation results for linear
        if len(linear_violations) > 0:
            linear_result = 'yes'  
        else:
            linear_result = "No"

        # Prepare the violation results for nonlinear
        if len(nonlinear_violations) > 0:
            nonlinear_result = 'yes'
        else:
            nonlinear_result = "No"

        # Append the results to the table data
        table_data.append([label, linear_result, nonlinear_result])

    # Define the table headers
    headers = ["Control Input", "Linear Violation", "Nonlinear Violation"]

    # Display the results in a nice tabular format
    print("\nControl Input Violation Summary:")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", numalign="center", stralign="center"))

def plot_kappa_over_time(kappa_total_linear, kappa_total_non_linear, tt):
    # Create the plot
    fig, ax5 = plt.subplots(figsize=(10, 6))

    # Plot kappa values for the linear model with solid line
    ax5.plot(tt[::10], kappa_total_linear[:len(tt[::10])], label='Kappas (Linear)', color='blue', linestyle=':')

    # Plot kappa values for the nonlinear model with dashed line
    ax5.plot(tt[::10], kappa_total_non_linear[:len(tt[::10])], label='Kappas (Nonlinear)', color='red', linestyle='--')

    # Set plot labels and title
    ax5.set_title('Kappas over Time for Linear and Nonlinear Models')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Kappa Value')

    # Add a legend
    ax5.legend()

    # Show the plot
    plt.show()
         
if __name__ == '__main__':
    # Simulate both models
    xx_linear, xc_linear, tt, kappa_total_linear = SRG_linear(model="linear")  # For linear model
    xx_non_linear, xc_non_linear, tt, kappa_total_non_linear= SRG_linear(model="nonlinear")  # For nonlinear model
    


    # Plot results for state variables side by side
    plot_xyz_side_by_side(tt, xx_linear, xx_non_linear)

    # Compare positions (Optional)
    compare_xyz_positions(tt, xx_linear, xx_non_linear)
   
    #plot conntrols and compare them.

    check_constraints(xc_linear,xc_non_linear)
    plot_control_inputs_side_by_side_try(tt, xc_linear, xc_non_linear)
    
  # Now call the plotting function
    plot_kappa_over_time(kappa_total_linear, kappa_total_non_linear, tt)
            

    