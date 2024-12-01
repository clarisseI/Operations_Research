import numpy as np
import matplotlib.pyplot as plt
from Constraints import *  # Assuming the necessary matrices like A, B, C, K, etc. are imported
from Figure_8_Trajectory import generate_waypoints_figure_8
from tabulate import tabulate
import numpy as np



def non_linear_quadrotor_euler(current_state, control):
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

    return np.array([u_dot, u, v_dot, v, w_dot, w, p_dot, p, q_dot, q, r_dot, r])

def linear_quadrotor_euler(current_state, control):
    """Computes the linear dynamics of the drone system."""
    return A@ current_state + B @ control

def simulate_euler(model_type="linear"):
    
    """Simulates both linear and nonlinear quadrotor dynamics."""
    #tt= np.arange( 0, T, Ts)
    Ts = 0.001
    T = 10
    tt= np.arange( 0, T+Ts, Ts)
    Ns= tt.size
    x0= np.zeros(12) #initial state
    xc=np.zeros(4) # control 
    ref= np.zeros(4) # reference 
    ref=[1,1,1,0]
    x_total= np.zeros((Ns, 12))
    cu_total= np.zeros((Ns, 4))
    for j in range(1, Ns):
        # Control law
        cu = -K @ x0 + Kc @ xc  # Control input
        if model_type == "linear":
            x0 = x0 + linear_quadrotor_euler(x0, cu) * Ts
        elif model_type == "nonlinear":
            x0 = x0 + non_linear_quadrotor_euler(x0, cu) * Ts

        x_total[j, :] = x0  # Store state
        cu_total[j,:]= cu
        #cu_total[j, :] = cu
        e = np.array([ref[0] - x0[1], ref[1] - x0[3], ref[2] - x0[5], ref[3] - x0[11]])
        xc = xc + e * Ts  # Update integral error

    return tt,x_total,cu_total

def plot_xyz_side_by_side(time_eval, x_total_linear, x_total_nonlinear):
    """Plots x, y, and z positions for linear and nonlinear models side by side."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Linear vs Nonlinear Dynamics", fontsize=16)

    labels = ["x-position", "y-position", "z-position"]
    colors = ["blue", "green", "red"]
    linear_titles = ["Linear Dynamics"] * 3
    nonlinear_titles = ["Nonlinear Dynamics"] * 3

    for i, (label, color, linear_title, nonlinear_title) in enumerate(zip(labels, colors, linear_titles, nonlinear_titles)):
        # Linear plots
        axes[i, 0].plot(time_eval, x_total_linear[:, i * 2 + 1], label=f"Linear {label}", color=color)
        axes[i, 0].set_title(linear_title, fontsize=12)
        axes[i, 0].set_ylabel(f"{label} (m)", fontsize=10)
        axes[i, 0].set_xlabel("Time (s)", fontsize=10)
        axes[i, 0].grid()
        axes[i, 0].legend()

        # Nonlinear plots
        axes[i, 1].plot(time_eval, x_total_nonlinear[:, i * 2 + 1], label=f"Nonlinear {label}", color=color)
        axes[i, 1].set_title(nonlinear_title, fontsize=12)
        axes[i, 1].set_ylabel(f"{label} (m)", fontsize=10)
        axes[i, 1].set_xlabel("Time (s)", fontsize=10)
        axes[i, 1].grid()
        axes[i, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def compare_xyz_positions(time_eval, x_total_linear, x_total_nonlinear):
    # Extract positions for X, Y, Z for both linear and nonlinear models
    x_linear = x_total_linear[:, 1]
    y_linear = x_total_linear[:, 3]
    z_linear = x_total_linear[:, 5]
    
    x_non_linear = x_total_nonlinear[:, 1]
    y_non_linear = x_total_nonlinear[:, 3]
    z_non_linear = x_total_nonlinear[:, 5]

    # Calculate absolute differences
    diff_x = np.abs(x_linear - x_non_linear)
    diff_y = np.abs(y_linear - y_non_linear)
    diff_z = np.abs(z_linear - z_non_linear)
    
    # Calculate average differences
    avg_diff_x = np.mean(diff_x)
    avg_diff_y = np.mean(diff_y)
    avg_diff_z = np.mean(diff_z)

    # Print average differences with formatting
    print(f"\n{'Average Absolute Differences':^40}")
    print("-" * 40)
    print(f"Average absolute difference for X: {avg_diff_x:.4f}")
    print(f"Average absolute difference for Y: {avg_diff_y:.4f}")
    print(f"Average absolute difference for Z: {avg_diff_z:.4f}")
    print("-" * 40)

    # Sample 10 values for better insight
    sample_indices = np.linspace(0, len(time_eval) - 1, 10, dtype=int)
    
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
    print("\n10 Sampled Positions for Linear and Nonlinear Trajectories:")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", numalign="center", stralign="center"))

def plot_control_inputs_side_by_side(time_eval, cu_total_linear, cu_total_nonlinear):
    """Plots the control inputs F, tau-phi, tau-theta, and tau-psi for linear and nonlinear models side by side."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle("Control Inputs (Linear vs Nonlinear) for F, Tau-Phi, Tau-Theta, Tau-Psi", fontsize=12)

    labels = ["F ", "Tau-Phi ", "Tau-Theta ", "Tau-Psi "]
    control_indices = [0, 1, 2, 3]  # F, Tau-Phi, Tau-Theta, Tau-Psi
 

    for i, (label, idx) in enumerate(zip(labels, control_indices)):
        # Plot linear control inputs
        axes[i, 0].plot(time_eval, cu_total_linear[:, idx], label=f"Linear {label}", color="blue")
        axes[i, 0].set_title(f"Linear {label}", fontsize=10)
        axes[i, 0].set_ylabel(f"{label} ", fontsize=8)
        axes[i, 0].set_xlabel("Time (s)", fontsize=8)
        axes[i, 0].grid()
        axes[i, 0].legend()

        # Plot nonlinear control inputs
        axes[i, 1].plot(time_eval, cu_total_nonlinear[:, idx], label=f"Nonlinear {label}", color="red")
        axes[i, 1].set_title(f"Nonlinear {label}", fontsize=10)
        axes[i, 1].set_ylabel(f"{label} ", fontsize=8)
        axes[i, 1].set_xlabel("Time (s)", fontsize=8)
        axes[i, 1].grid()
        axes[i, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def check_constraints(time_eval, cu_total_linear, cu_total_nonlinear):
    """Check if the control inputs F, Tau-Phi, Tau-Theta, and Tau-Psi violate the constraints."""
    labels = ["F", "Tau-Phi", "Tau-Theta", "Tau-Psi"]
    control_indices = [0, 1, 2, 3]  # F, Tau-Phi, Tau-Theta, Tau-Psi
    constraints_min = [-6, -0.05, -0.05, -0.05]
    constraints_max = [6, 0.005, 0.005, 0.005]

    table_data = []

    for label, idx in zip(labels, control_indices):
        # Find the indices where the control inputs violate the constraints
        linear_violations = np.where((cu_total_linear[:, idx] < constraints_min[idx]) | (cu_total_linear[:, idx] > constraints_max[idx]))[0]
        nonlinear_violations = np.where((cu_total_nonlinear[:, idx] < constraints_min[idx]) | (cu_total_nonlinear[:, idx] > constraints_max[idx]))[0]

        # Prepare the violation results for linear
        if len(linear_violations) > 0:
            linear_result = f"Yes, range: {time_eval[linear_violations[0]]} to {time_eval[linear_violations[-1]]}"
        else:
            linear_result = "No"

        # Prepare the violation results for nonlinear
        if len(nonlinear_violations) > 0:
            nonlinear_result = f"Yes, range: {time_eval[nonlinear_violations[0]]} to {time_eval[nonlinear_violations[-1]]}"
        else:
            nonlinear_result = "No"

        # Append the results to the table data
        table_data.append([label, linear_result, nonlinear_result])

    # Define the table headers
    headers = ["Control Input", "Linear Violation", "Nonlinear Violation"]

    # Display the results in a nice tabular format
    print("\nControl Input Violation Summary:")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", numalign="center", stralign="center"))

def simulate_figure_8_euler(model_type="linear"):
    
   # Generate time and waypoints for figure-8 trajectory
    waypoints = generate_waypoints_figure_8(x_amplitude, y_amplitude, omega, z0, steps)  
    Ts = 0.001
    T = 500
    tt = np.arange(0, T, Ts)
    Ns = tt.size
    x0 = np.zeros(12)  # Initial state
    xc = np.zeros(4)   # Control state
    x_total = np.zeros((Ns, 12))
    cu_total = np.zeros((Ns, 4))
    y = 0  # Index for waypoints
    len_waypoints = waypoints.shape[0]  # Total number of waypoints

    # Create lists to store the trajectory in 3D
    x_traj, y_traj, z_traj = [], [], []
    
    for j in range(0, Ns):
        # Control law
        cu = -K @ x0 + Kc @ xc  # Control input
        if model_type == "linear":
            x0 = x0 + linear_quadrotor_euler(x0, cu) * Ts
        elif model_type == "nonlinear":
            x0 = x0 + non_linear_quadrotor_euler(x0, cu) * Ts
        x_total[j, :] = x0  # Store the state
        cu_total[j, :] = cu  # Store control input

        # Calculate position error (e) based on the current waypoint
        e = np.array([
            waypoints[y, 1] - x0[1],  # x-position error
            waypoints[y, 3] - x0[3],  # y-position error
            waypoints[y, 5] - x0[5],  # z-position error
            waypoints[y, 11] - x0[11]  # yaw error (psi)
        ])

        # Update the integral error (xc)
        xc = xc + e * Ts

        # Calculate distance between the current state and the waypoint
        dist = np.sqrt(
            (x0[1] - waypoints[y, 1]) ** 2 + 
            (x0[3] - waypoints[y, 3]) ** 2 + 
            (x0[5] - waypoints[y, 5]) ** 2
        )

        # If the distance is less than 0.01, move to the next waypoint
        if dist <= 0.1:
            
            # Store current position for plotting
            x_traj.append(x0[1])  # x position
            y_traj.append(x0[3])  # y position
            z_traj.append(x0[5])  # z position
            y += 1  # Move to the next waypoint
            # Ensure waypoint_t doesn't go out of bounds
            if y >= len_waypoints:
                y = len_waypoints - 1  # Stay at the last waypoint
        #print ("distance",dist)
    
    return tt, x_total, x_traj, y_traj, z_traj, waypoints, cu_total

def plot_3d_trajectory(x_traj, y_traj, z_traj, waypoints):
    """Plot the 3D trajectory and waypoints."""
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

   # Plot the drone trajectory as a solid red line
    ax.plot(x_traj, y_traj, z_traj, label="Drone Trajectory", color="red",linestyle="dashdot", linewidth=2)

    # Plot the waypoints as a semi-transparent dashed blue line
    ax.plot(waypoints[:, 1], waypoints[:, 3], waypoints[:, 5], 
            color="blue", linestyle="--", linewidth=1, alpha=0.7, label="Waypoints")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Drone Trajectory vs Waypoints")
    ax.legend()

    plt.show()

# Example usage:
# compare_xyz_positions(sol_linear, sol_non_linear, time_eval)

# Example usage
if __name__ == "__main__":
    # Simulate both models
    tt, x_total_linear, cu_total_linear = simulate_euler(model_type="linear")
    tt, x_total_nonlinear, cu_total_nonlinear = simulate_euler(model_type="nonlinear")

    # Plot results for state variables side by side
    plot_xyz_side_by_side(tt, x_total_linear, x_total_nonlinear)

   # Plot control inputs for both linear and nonlinear models side by side
    plot_control_inputs_side_by_side(tt, cu_total_linear, cu_total_nonlinear)
    
    # Compare positions (Optional)
    compare_xyz_positions(tt, x_total_linear, x_total_nonlinear)
    
    # Check constraints violation for both Linear and Non linear
    check_constraints(tt, cu_total_linear, cu_total_nonlinear)
    
    tt, x_total_linear_figure_8, x_traj_linear_figure_8, y_traj_linear_figure_8, z_traj_linear_figure_8, waypoints_linear_figure_8,cu_total_linear_figure_8 = simulate_figure_8_euler(model_type='linear')
    tt, x_total_nonlinear_figure_8, x_traj_nonlinear_figure_8, y_traj_nonlinear_figure_8, z_traj_nonlinear_figure_8, waypoints_nonlinear_figure_8, cu_total_nonlinear_figure_8 = simulate_figure_8_euler(model_type='nonlinear')
    plot_3d_trajectory(x_traj_linear_figure_8, y_traj_linear_figure_8, z_traj_linear_figure_8, waypoints_linear_figure_8)
    plot_3d_trajectory(x_traj_nonlinear_figure_8, y_traj_nonlinear_figure_8, z_traj_nonlinear_figure_8, waypoints_nonlinear_figure_8)
    
    # Check constraints violation for both linear and non linear for Figure _8
    check_constraints(tt, cu_total_linear_figure_8, cu_total_nonlinear_figure_8)
    
    # Plot control inputs for both linear and nonlinear models side by side for Figure_8
    plot_control_inputs_side_by_side(tt, cu_total_linear_figure_8, cu_total_nonlinear_figure_8)
    
    
    



