import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Constraints import *
import pandas as pd
from Figure_8_Trajectory import generate_waypoints_figure_8

class IntegralControl:
    def __init__(self):
        self.control_values_linear = {
            'linear': {'F': [], 'tau_phi': [], 'tau_theta': [], 'tau_psi': []},
            'non_linear': {'F': [], 'tau_phi': [], 'tau_theta': [], 'tau_psi': []}
            
        }
        

    def integral_control_linear(self, time, current_state, target_state):
            
        target_state = np.array(target_state)
        current_state = np.array(current_state)

        # Getting 4x1 Integral u
        error = (target_state[[1, 3, 5, 11]] - current_state[[1, 3, 5, 11]])
        control = -K @ current_state + Kc @ error
         #Store control values
        self.control_values_linear['linear']['F'].append(float(control[0]) )         # 'F'
        self.control_values_linear['linear']['tau_phi'].append(float(control[1]) )  # 'tau_phi'
        self.control_values_linear['linear']['tau_theta'].append(float(control[2])) #  'tau_theta'
        self.control_values_linear['linear']['tau_psi'].append(float(control[3]) ) #  'tau_psi'
            
        dx = A @ current_state + B @ control
        return dx
    
    def integral_control_non_linear(self, time, current_state, target_state):
        u, px, v, py, w, pz, p, phi, q, theta, r, psi = current_state
       
        
        target_state = np.array(target_state)
        current_state = np.array(current_state)

        # Getting 4x1 Integral u
        error = (target_state[[1, 3, 5, 11]] - current_state[[1, 3, 5, 11]])
        control = -K @ current_state + Kc @ error
        control[0] += M * g  # Adding gravity effect to z-axis thrust

        # Thrust force
        F = control[0]
         #Store control values
        self.control_values_linear['non_linear']['F'].append(float(F) )         # 'F'
        self.control_values_linear['non_linear']['tau_phi'].append(float(control[1]) )  # 'tau_phi'
        self.control_values_linear['non_linear']['tau_theta'].append(float(control[2])) #  'tau_theta'
        self.control_values_linear['non_linear']['tau_psi'].append(float(control[3]) ) #  'tau_psi'

        # Linear velocity dynamics
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

    def compare_control_values(self):
        # Define the constraints
        constraints = {
            'F': 6,           # F <= 6 and -F <= 6
            'tau_phi': (0.005, 0.05),  # 0.005 <= tau_phi <= 0.05
            'tau_theta': (0.005, 0.05),  # 0.005 <= tau_theta <= 0.05
            'tau_psi': (0.005, 0.05),  # 0.005 <= tau_psi <= 0.05
        }

        # Initialize dictionaries to store whether there are violations for both 'linear' and 'non_linear'
        violations = {
            'linear': {
                'F': False,
                'tau_phi': False,
                'tau_theta': False,
                'tau_psi': False
            },
            'non_linear': {
                'F': False,
                'tau_phi': False,
                'tau_theta': False,
                'tau_psi': False
            }
        }

        # Check violations for 'linear' control values
        for key in self.control_values_linear['linear']:
            for value in self.control_values_linear['linear'][key]:
                value = float(value)  # Convert np.float to regular float
                if key == 'F':
                    if abs(value) > constraints['F']:  # F <= 6 and -F <= 6
                        violations['linear']['F'] = True
                elif key in ['tau_phi', 'tau_theta', 'tau_psi']:
                    min_val, max_val = constraints[key]
                    if not (min_val <= abs(value) <= max_val):  # Check if within range [min, max]
                        violations['linear'][key] = True

        # Check violations for 'non_linear' control values
        for key in self.control_values_linear['non_linear']:
            for value in self.control_values_linear['non_linear'][key]:
                value = float(value)  # Convert np.float to regular float
                if key == 'F':
                    if abs(value) > constraints['F']:  # F <= 6 and -F <= 6
                        violations['non_linear']['F'] = True
                elif key in ['tau_phi', 'tau_theta', 'tau_psi']:
                    min_val, max_val = constraints[key]
                    if not (min_val <= abs(value) <= max_val):  # Check if within range [min, max]
                        violations['non_linear'][key] = True

        return violations

    def solve_integral_control(self, time_span, initial_state, target_state):
       

        sol_linear_integral = solve_ivp(self.integral_control_linear, time_span, initial_state,args=(target_state,), dense_output=True)
        sol_non_linear_integral = solve_ivp(self.integral_control_non_linear, time_span, initial_state,
                                            args=(target_state,), dense_output=True)
        
        return sol_linear_integral,sol_non_linear_integral
    
    def plot_xyz_from_ivp_solution(self, time_eval, sol, title_prefix="Non-Linear Integral Control"):
        """Plots x, y, and z positions from the solve_ivp solution on separate graphs."""
        # Interpolate solution at desired times
        interpolated_states = sol.sol(time_eval)  # Shape: (state_dim, len(time_eval))

        # Extract x, y, z positions (assuming indices for px, py, pz are 1, 3, and 5 respectively)
        x_position = interpolated_states[1, :]
        y_position = interpolated_states[3, :]
        z_position = interpolated_states[5, :]

        # Create subplots
        plt.figure(figsize=(12, 8))

        # x-position plot
        plt.subplot(3, 1, 1)
        plt.plot(time_eval, x_position, label="x-position (px)", color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("x (m)")
        plt.grid()
        plt.legend()

        # y-position plot
        plt.subplot(3, 1, 2)
        plt.plot(time_eval, y_position, label="y-position (py)", color="green")
        plt.xlabel("Time (s)")
        plt.ylabel("y (m)")
        plt.title(f"{title_prefix} - y-Position")
        plt.grid()
        plt.legend()

        # z-position plot
        plt.subplot(3, 1, 3)
        plt.plot(time_eval, z_position, label="z-position (pz)", color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("z (m)")
        plt.title(f"{title_prefix} - z-Position")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

    def compare_xyz_positions(self, sol_linear, sol_non_linear, time_eval):
        linear_states = sol_linear.sol(time_eval)
        non_linear_states = sol_non_linear.sol(time_eval)
        
         # Extract positions
        x_linear = linear_states[1, :]
        y_linear = linear_states[3, :]
        z_linear = linear_states[5, :]

        x_non_linear = non_linear_states[1, :]
        y_non_linear = non_linear_states[3, :]
        z_non_linear = non_linear_states[5, :]

         # Calculate absolute differences
        diff_x = np.abs(x_linear - x_non_linear)
        diff_y = np.abs(y_linear - y_non_linear)
        diff_z = np.abs(z_linear - z_non_linear)
        
         # Calculate average differences
        avg_diff_x = np.mean(diff_x)
        avg_diff_y = np.mean(diff_y)
        avg_diff_z = np.mean(diff_z)
        # Print average differences
        print(f"Average absolute difference for X: {avg_diff_x:.4f}")
        print(f"Average absolute difference for Y: {avg_diff_y:.4f}")
        print(f"Average absolute difference for Z: {avg_diff_z:.4f}")

         # Sample 10 values
        sample_indices = np.linspace(0, len(time_eval) - 1, 10, dtype=int)

        print("\n10 Sampled Positions for Linear and Nonlinear Trajectories:")
        print(f"{'Time (s)':<10} {'X_Linear':<10} {'Y_Linear':<10} {'Z_Linear':<10} {'X_Nonlinear':<12} {'Y_Nonlinear':<12} {'Z_Nonlinear':<12}")
        print("-" * 70)
        for i in sample_indices:
            print(
                f"{time_eval[i]:<10.2f} {x_linear[i]:<10.4f} {y_linear[i]:<10.4f} {z_linear[i]:<10.4f} "
                f"{x_non_linear[i]:<12.4f} {y_non_linear[i]:<12.4f} {z_non_linear[i]:<12.4f}"
            )
    
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


    def plot_control_values(self, control_type='linear', time_eval=None):
        # Ensure time_eval is provided
        if time_eval is None:
            raise ValueError("time_eval must be provided for plotting.")
        
        # Check if control_type is valid (either 'linear' or 'non_linear')
        if control_type not in ['linear', 'non_linear']:
            raise ValueError("control_type must be 'linear' or 'non_linear'")
        
        # Extract the control values based on the control type
        F = self.control_values_linear[control_type]['F']
        tau_phi = self.control_values_linear[control_type]['tau_phi']
        tau_theta = self.control_values_linear[control_type]['tau_theta']
        tau_psi = self.control_values_linear[control_type]['tau_psi']
        # Downsample control values to match the length of time_eval
        F = F[:len(time_eval)]
        tau_phi = tau_phi[:len(time_eval)]
        tau_theta = tau_theta[:len(time_eval)]
        tau_psi = tau_psi[:len(time_eval)]
        
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot F
        axs[0, 0].plot(time_eval, F, label=f'{control_type.capitalize()} F', color='blue')
        axs[0, 0].set_title('F Control Value')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('F')
        axs[0, 0].legend()

        # Plot tau_phi
        axs[0, 1].plot(time_eval, tau_phi, label=f'{control_type.capitalize()} tau_phi', color='blue')
        axs[0, 1].set_title('tau_phi Control Value')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('tau_phi')
        axs[0, 1].legend()

        # Plot tau_theta
        axs[1, 0].plot(time_eval, tau_theta, label=f'{control_type.capitalize()} tau_theta', color='blue')
        axs[1, 0].set_title('tau_theta Control Value')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('tau_theta')
        axs[1, 0].legend()

        # Plot tau_psi
        axs[1, 1].plot(time_eval, tau_psi, label=f'{control_type.capitalize()} tau_psi', color='blue')
        axs[1, 1].set_title('tau_psi Control Value')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('tau_psi')
        axs[1, 1].legend()

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()


   
if __name__ == "__main__":
    # Example Usage
    integral_control = IntegralControl()
    initial_state = np.zeros(12)  # Example initial state
    target_state = np.zeros(12)  # Example target state
    target_state= [0,1,0,1,0,1,0,0,0,0,0,0] # Example target positions for x, y, z

    time_span = (0, 10)
    time_eval = np.linspace(0, 10, 100)
    # Example usage
    sol_linear_integral, sol_non_linear_integral= integral_control.solve_integral_control(time_span, initial_state, target_state)
    # Assuming `sol_non_linear_integral` is the solution obtained from solve_ivp
    integral_control.plot_xyz_from_ivp_solution(time_eval, sol_non_linear_integral, title_prefix="Non-Linear Integral Control")
    # Assuming `sol_non_linear_integral` is the solution obtained from solve_ivp
    integral_control.plot_xyz_from_ivp_solution(time_eval, sol_linear_integral, title_prefix="Non-Linear Integral Control")
       
    comparison_table = integral_control.compare_xyz_positions(sol_linear_integral, sol_non_linear_integral, time_eval)
    print(comparison_table)
    
    violations= integral_control.compare_control_values()
     # Print the violations for both linear and non_linear controls
    print("Linear Control Violations:", violations['linear'])
    print("Non-linear Control Violations:", violations['non_linear'])
    
    # Generate time and waypoints for figure-8 trajectory
    t_eval, waypoints = generate_waypoints_figure_8(x_amplitude, y_amplitude, omega, z0, steps)  
    # Generate figure-8 trajectory for linear model
    x_fig8, y_fig8, z_fig8 = [], [], []
    for i in range(len(time_eval-1)):
            current_state = waypoints[i-1]
            target_state = waypoints[i]
        
            # Set up time interval for integration (e.g., from t[i] to t[i+1])
            t_interval = (time_eval[i-1], time_eval[i])
        
            # Run solve_ivp over this interval
            sol_figure_8 = solve_ivp(integral_control.integral_control_linear, t_interval, current_state, args=(target_state,))
            
            # Collect the solution points for x, y, z
            x_fig8.extend(sol_figure_8.y[1])  # Assuming x is at index 1
            y_fig8.extend(sol_figure_8.y[3])  # Assuming y is at index 3
            z_fig8.extend(sol_figure_8.y[5])  # Assuming z is at index 5

    
    integral_control.plot_figure_8_trajectory( x_fig8, y_fig8, z_fig8)
    
    # Plot the linear control values
    integral_control.plot_control_values(control_type='linear', time_eval= time_span)

    # Plot the non-linear control values
    integral_control.plot_control_values(control_type='non_linear', time_eval=time_span)