import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Constraints import *
import pandas as pd
from Figure_8_Trajectory import generate_waypoints_figure_8

# Define the LQR controller class
class FeedForwardControl:
    #Class for LQR controller design for both linear and non-linear Model
    
    def __init__(self):
        self.control_values = {
            'linear': {'F': [], 'tau_phi': [], 'tau_theta': [], 'tau_psi': [], 'time_steps': []},
            'non_linear': {'F': [], 'tau_phi': [], 'tau_theta': [], 'tau_psi': [], 'time_steps': []}
        }
    
    def compute_control(self, current_state, target_state, model_type):
        """Computes control values based on LQR feedback law."""
        control = -K @ (current_state - target_state)
        if model_type == 'non_linear':
            control[0] += M * g  # Add gravity compensation for z-axis thrust
        return control
    
    def feedforward_non_linear(self, time, current_state, target_state):
        # the current state
        u, px, v, py, w, pz, p, phi, q, theta, r, psi = current_state

        # Control law (feedback control based on error)
        control = - K @ (current_state - target_state)

        # Adding gravity effect to the control for z-axiz thrust
        control[0] += M * g
    
        # Thrust force
        F = control[0]

        #Store control values
        self.store_control_values('non_linear', time, control)
        
        # Linear velocity dynamics (u_dot, v_dot, w_dot)
        u_dot = r*v - q*w - g*np.sin(theta)
        v_dot = p*w - r*u + g*np.cos(theta)*np.sin(phi)
        w_dot = q*u - p*v + g*np.cos(theta)*np.cos(phi) - F/M

        # Angular velocity dynamics (p_dot, q_dot, r_dot)
        p_dot = (Jy - Jz) / Jx * q * r + control[1] / Jx  # control[1] corresponds to torque τ_φ
        q_dot = (Jz - Jx) / Jy * p * r + control[2] / Jy  # control[2] corresponds to torque τ_θ
        r_dot = (Jx - Jy) / Jz * p * q + control[3] / Jz  # control[3] corresponds to torque τ_ψ

        #return the derivatives
        dx = np.array([u_dot,u,v_dot,v,w_dot,w, p_dot,p, q_dot,q,r_dot,r])

        return dx
    
    def feedforward_linear(self,time,current_state, A,B, target_state):
        """
        Computes the linear dynamics of the drone system.
        Returns dx
        """
        # Control law (feedback control based on state error)
        control = -K @ (current_state - target_state)
        
        # Linear state update: dx = Ax + B * control
        dx = A@ current_state + B@ control
        
        # Store control values
        self.store_control_values('linear', time, control)
    
        return dx
    
    def solve_feedforward_control(self, initial_state, target_state, time_span,A=None, B=None):
        
    
        # Solve nonlinear dynamics
        sol_feedforward_non_linear = solve_ivp(self.feedforward_non_linear, time_span, initial_state, args=(target_state,),dense_output=True)
    
        # Solve linear dynamics
        sol_feedforward_linear = solve_ivp(self.feedforward_linear, time_span, initial_state, args=(A, B,target_state,), dense_output=True)
        
    
        return sol_feedforward_non_linear, sol_feedforward_linear
    
    def store_control_values(self, model_type, time, control):
        """Store control values in the control_values dictionary."""
        if model_type not in self.control_values:
            print(f"Error: {model_type} is not a valid model type.")
            return

        print(f"Storing values for {model_type} at time {time}: {control}")  # Debugging print

        self.control_values[model_type]['F'].append(control[0])  # For force (thrust)
        self.control_values[model_type]['tau_phi'].append(control[1])  # For phi torque
        self.control_values[model_type]['tau_theta'].append(control[2])  # For theta torque
        self.control_values[model_type]['tau_psi'].append(control[3])  # For psi torque
        self.control_values[model_type]['time_steps'].append(time)  # For time steps
        
    def plot_controls(self, control_labels, model_types):
        """Plots control inputs for specified model types."""
        fig, axs = plt.subplots(len(control_labels), len(model_types), figsize=(14, 12))
        
        for i, label in enumerate(control_labels):
            for j, model_type in enumerate(model_types):
                data = self.control_values[model_type][label]
                axs[i, j].plot(self.control_values[model_type]['time_steps'], data, label=f"{model_type.capitalize()} {label}")
                axs[i, j].set_title(f"{model_type.capitalize()} {label}")
                axs[i, j].set_xlabel("Time (s)")
                axs[i, j].set_ylabel(label)
                axs[i, j].legend()
                axs[i, j].grid()
        
        plt.tight_layout()
        plt.show()

    def check_constraints(self):
        """Check if control values violate constraints for both linear and nonlinear models."""
        # Define min and max constraints for control values
        min_force, max_force = -6, 6  # Min and Max for thrust force (F)
        min_phi, max_phi = -0.005, 0.005  # Min and Max for torque (tau_phi, tau_theta, tau_psi)
        min_theta, max_theta = -0.005, 0.005
        min_psi, max_psi = -0.005, 0.005

        # Iterate through both models (linear and non_linear)
        for model_type in ['linear', 'non_linear']:
            control_values = self.control_values[model_type]
            
            # Debugging: Check length of each control value list
            print(f"Checking {model_type} model. Control data lengths: "
                f"Force: {len(control_values['F'])}, Phi: {len(control_values['tau_phi'])}, "
                f"Theta: {len(control_values['tau_theta'])}, Psi: {len(control_values['tau_psi'])}")
            
            any_violation = False  # Flag to track if any violation occurs
            
            # Check each control value for violations
            for i in range(len(control_values['F'])):
                # Get control values for the current time step
                force = control_values['F'][i]
                tau_phi = control_values['tau_phi'][i]
                tau_theta = control_values['tau_theta'][i]
                tau_psi = control_values['tau_psi'][i]
                time = control_values['time_steps'][i]

                # Debugging: Print values being checked
                print(f"Checking model: {model_type} | Time: {time:.4f} | Force: {force:.4f} | "
                    f"Phi: {tau_phi:.4f} | Theta: {tau_theta:.4f} | Psi: {tau_psi:.4f}")

                # Initialize violation flags
                violations = []

                # Check for force violation first
                if not (min_force <= force <= max_force):
                    violations.append(f"Force Violated at {force:.4f}")
                    print(f"Violation found at time {time:.4f} for Force")  # Debugging
                    break  # Exit the inner loop if force violation is found

                # If no force violation, check other controls
                if not (min_phi <= tau_phi <= max_phi):
                    violations.append(f"Phi Violated at {tau_phi:.4f}")
                if not (min_theta <= tau_theta <= max_theta):
                    violations.append(f"Theta Violated at {tau_theta:.4f}")
                if not (min_psi <= tau_psi <= max_psi):
                    violations.append(f"Psi Violated at {tau_psi:.4f}")

                # If any violation occurs, print it and set the flag
                if violations:
                    any_violation = True
                    print(f"{model_type.capitalize()} Model Violation at Time: {time:.4f}")
                    for violation in violations:
                        print(f"  - {violation}")
                    break  # Exit the loop for the current time step after printing violations

            # After checking all time steps for the current model, print if no violations were found
            if not any_violation:
                print(f"{model_type.capitalize()} Model: No Violations at any time step")

                  



    def plot_states(self, time, states, labels, title_prefix=""):
        """Generic function to plot states over time."""
        num_states = len(labels)
        fig, axs = plt.subplots(num_states, 1, figsize=(12, 4 * num_states))
        
        for i, label in enumerate(labels):
            axs[i].plot(time, states[:, i], label=label, color='b')
            axs[i].set_title(f"{title_prefix} {label}")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel(label)
            axs[i].legend()
            axs[i].grid()
        
        plt.tight_layout()
        plt.show()

    def plot_figure_8_trajectory(self, x_fig8_l, y_fig8_l, z_fig8_l):
        """
        Plot and compare the nonlinear and linear figure-8 trajectories.
        """
        fig_3d = plt.figure(figsize=(14, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
     
        ax_3d.plot(x_fig8_l, y_fig8_l, z_fig8_l, label="Linear Trajectory", color="red")

        ax_3d.set_xlabel("X Position (m)")
        ax_3d.set_ylabel("Y Position (m)")
        ax_3d.set_zlabel("Z Position (m)")
        ax_3d.set_title("Figure-8 Trajectory Comparison")
        ax_3d.legend()

        plt.show()

        
    def compare_xyz_positions(self, sol_feedforward_non_linear, sol_feedforward_linear, t_eval):
        # Interpolate solutions
        nonlinear_states = sol_feedforward_non_linear.sol(t_eval)
        linear_states = sol_feedforward_linear.sol(t_eval)

        # Extract positions
        x_nonlinear = nonlinear_states[1, :]
        y_nonlinear = nonlinear_states[3, :]
        z_nonlinear = nonlinear_states[5, :]

        x_linear = linear_states[1, :]
        y_linear = linear_states[3, :]
        z_linear = linear_states[5, :]

        # Calculate absolute differences
        diff_x = np.abs(x_linear - x_nonlinear)
        diff_y = np.abs(y_linear - y_nonlinear)
        diff_z = np.abs(z_linear - z_nonlinear)

        # Calculate average differences
        avg_diff_x = np.mean(diff_x)
        avg_diff_y = np.mean(diff_y)
        avg_diff_z = np.mean(diff_z)

        # Print average differences
        print(f"Average absolute difference for X: {avg_diff_x:.4f}")
        print(f"Average absolute difference for Y: {avg_diff_y:.4f}")
        print(f"Average absolute difference for Z: {avg_diff_z:.4f}")

        # Sample 10 values
        sample_indices = np.linspace(0, len(t_eval) - 1, 10, dtype=int)

        print("\n10 Sampled Positions for Linear and Nonlinear Trajectories:")
        print(f"{'Time (s)':<10} {'X_Linear':<10} {'Y_Linear':<10} {'Z_Linear':<10} {'X_Nonlinear':<12} {'Y_Nonlinear':<12} {'Z_Nonlinear':<12}")
        print("-" * 70)
        for i in sample_indices:
            print(
                f"{t_eval[i]:<10.2f} {x_linear[i]:<10.4f} {y_linear[i]:<10.4f} {z_linear[i]:<10.4f} "
                f"{x_nonlinear[i]:<12.4f} {y_nonlinear[i]:<12.4f} {z_nonlinear[i]:<12.4f}"
            )

   


        
if __name__ == "__main__":
     # Define initial parameters
    initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target_state = [0, 1, 0, 1, 0, 10, 0, 0, 0, 0, 0, 0]
    time_span = [0, 10]
    t_eval = np.linspace(0, 10, 100)
    
    # Initialize controller
    feedforward = FeedForwardControl()
    
    # Solve dynamics
    sol_non_linear, sol_linear = feedforward.solve_feedforward_control(initial_state, target_state, time_span, A, B)
    
    # Plot states
    labels = ["x", "y", "z"]
    feedforward.plot_states(t_eval, sol_non_linear.sol(t_eval).T[:, [1, 3, 5]], labels, title_prefix="Nonlinear")
    feedforward.plot_states(t_eval, sol_linear.sol(t_eval).T[:, [1, 3, 5]], labels, title_prefix="Linear")
    
    # Plot controls
    control_labels = ['F', 'tau_phi', 'tau_theta', 'tau_psi']
    feedforward.plot_controls(control_labels, model_types=['linear', 'non_linear'])
    
    # Generate time and waypoints for figure-8 trajectory
    t_eval, waypoints = generate_waypoints_figure_8(x_amplitude, y_amplitude, omega, z0, steps)

    
   
     # Generate comparison table
    comparison_table = feedforward.compare_xyz_positions(sol_non_linear,sol_linear,t_eval)
    
    print(comparison_table)
   
    
    # Generate figure-8 trajectory for linear model
    x_fig8, y_fig8, z_fig8 = [], [], []
    for i in range(len(t_eval)-1):
            current_state = waypoints[i]
            target_state = waypoints[i+1]
        
            # Set up time interval for integration (e.g., from t[i] to t[i+1])
            t_interval = (t_eval[i], t_eval[i+1])
        
            # Run solve_ivp over this interval
            sol_figure_8 = solve_ivp(feedforward.feedforward_linear, t_interval, current_state, args=(A, B, target_state))
            
            # Collect the solution points for x, y, z
            x_fig8.extend(sol_figure_8.y[1])  # Assuming x is at index 1
            y_fig8.extend(sol_figure_8.y[3])  # Assuming y is at index 3
            z_fig8.extend(sol_figure_8.y[5])  # Assuming z is at index 5

    
    feedforward.plot_figure_8_trajectory( x_fig8, y_fig8, z_fig8)
   


    # Check constraints and print violations for each control
    feedforward.check_constraints()  # Will print each violated control for every time step