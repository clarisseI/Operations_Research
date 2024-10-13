import numpy as np
import matplotlib.pyplot as plt

# Import the necessary functions and constants from your existing code
from dynamic_operation import solve_quadrotor_dynamics

def plot_quadrotor_trajectories(sol_non_linear, sol_linear, target_points, time_spans):
    fig, axs = plt.subplots(len(target_points), 2, figsize=(14, 5 * len(target_points)))

    # Adjust layout for better visibility
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, (target_state, t_span) in enumerate(zip(target_points, time_spans)):
        time_eval = np.linspace(t_span[0], t_span[1], 500)  # Generate time points
        
        # Extracting positions for non-linear and linear solutions with increased offsets
        x_non_linear = sol_non_linear[i].y[1, :] + i * 0.3  # Increased offset for visibility
        y_non_linear = sol_non_linear[i].y[3, :] + i * 0.3  # Increased offset for visibility
        z_non_linear = sol_non_linear[i].y[5, :] + i * 0.3  # Increased offset for visibility

        x_linear = sol_linear[i].y[1, :] + i * 0.3  # Increased offset for visibility
        y_linear = sol_linear[i].y[3, :] + i * 0.3  # Increased offset for visibility
        z_linear = sol_linear[i].y[5, :] + i * 0.3  # Increased offset for visibility

        # Plot non-linear trajectories
        axs[i, 0].plot(time_eval, x_non_linear, label='x (Non-linear)', alpha=0.7)  # Added alpha for transparency
        axs[i, 0].plot(time_eval, y_non_linear, label='y (Non-linear)', alpha=0.7)
        axs[i, 0].plot(time_eval, z_non_linear, label='z (Non-linear)', alpha=0.7)

        axs[i, 0].set_title(f'Non-linear Quadrotor Trajectory (Target: {target_state})')
        axs[i, 0].set_xlabel('Time (seconds)')
        axs[i, 0].set_ylabel('Position (meters)')
        axs[i, 0].legend()
        axs[i, 0].grid(True)

        # Plot linear trajectories
        axs[i, 1].plot(time_eval, x_linear, label='x (Linear)', linestyle='--', alpha=0.7)  # Added alpha for transparency
        axs[i, 1].plot(time_eval, y_linear, label='y (Linear)', linestyle='--', alpha=0.7)
        axs[i, 1].plot(time_eval, z_linear, label='z (Linear)', linestyle='--', alpha=0.7)

        axs[i, 1].set_title(f'Linear Quadrotor Trajectory (Target: {target_state})')
        axs[i, 1].set_xlabel('Time (seconds)')
        axs[i, 1].set_ylabel('Position (meters)')
        axs[i, 1].legend()
        axs[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    initial_state= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Define target points and time spans for testing
    target_points = [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Target at [0, 0, 1]
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # Target at [1, 1, 1]
        [0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0],  # Target at [2, 2, 1]
        [0, -1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0] # Target at [-1, -1, 1]
    ]

    time_spans = [
        [0, 5],   # Simulate for 5 seconds
        [0, 10],  # Simulate for 10 seconds
        [0, 7],   # Simulate for 7 seconds
        [0, 3]    # Simulate for 3 seconds
    ]
    
    # Initialize lists to hold solutions for each target point
    sol_non_linear = []
    sol_linear = []

    # Solve dynamics for each target point
    for target_state, t_span in zip(target_points, time_spans):
        time_eval = np.linspace(t_span[0], t_span[1], 500)  # Generate time points
        sol_nl, sol_l = solve_quadrotor_dynamics(initial_state, target_state, t_span, time_eval)
        sol_non_linear.append(sol_nl)
        sol_linear.append(sol_l)

    # Plot the results
    plot_quadrotor_trajectories(sol_non_linear, sol_linear, target_points, time_spans)
