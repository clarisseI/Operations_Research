import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Constraints import *
from Figure_8_Trajectory import generate_waypoints_figure_8
# Define the LQR controller class
class LQRController:
    #Class for LQR controller design for both linear and non-linear Model
    def __init__(self):
        pass
      
      
    def lqr_non_linear(self, time, current_state, target_state):
        # the current state
        u, px, v, py, w, pz, p, phi, q, theta, r, psi = current_state

        # Control law (feedback control based on error)
        control = - K @ (current_state - target_state)

        # Adding gravity effect to the control for z-axiz thrust
        control[0] += M * g
    
        # Thrust force
        F = control[0]

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
    
    def lqr_linear(self,time,current_state, A,B, target_state):
        """
        Computes the linear dynamics of the drone system.
        Returns dx
        """
        # Control law (feedback control based on state error)
        control = -K @ (current_state - target_state)
        # Linear state update: dx = Ax + B * control
        dx = A@ current_state + B@ control
    
        return dx
    

    
    
    def solve_lqr_models(self, initial_state, target_state, time_span):
    
        # Solve nonlinear dynamics
        sol_lqr_non_linear = solve_ivp(self.lqr_non_linear, time_span, initial_state, args=(target_state,))
    
        # Solve linear dynamics
        sol_lqr_linear = solve_ivp(self.lqr_non_linear, time_span, initial_state, args=(target_state,))
        
    
        return sol_lqr_non_linear, sol_lqr_linear
        
    
    def plot_trajectories(self, sol_lqr_non_linear, sol_lqr_linear, x_fig8, y_fig8, z_fig8):
        
        fig = plt.figure(figsize=(18, 6))

        # Extract x, y, z positions from sol_lqr_non_linear and sol_lqr_linear
        x_nonlinear = sol_lqr_non_linear.y[1]  # Assuming x is at index 1
        y_nonlinear = sol_lqr_non_linear.y[3]  # Assuming y is at index 3
        z_nonlinear = sol_lqr_non_linear.y[5]  # Assuming z is at index 5
        
        x_linear = sol_lqr_linear.y[1]  # Assuming x is at index 1
        y_linear = sol_lqr_linear.y[3]  # Assuming y is at index 3
        z_linear = sol_lqr_linear.y[5]  # Assuming z is at index 5

        # Data and labels for each subplot
        trajectories = [
            {"x": x_nonlinear, "y": y_nonlinear, "z": z_nonlinear, "title": "LQR Nonlinear Model", "color": "blue", "subplot": 131},
            {"x": x_linear, "y": y_linear, "z": z_linear, "title": "LQR Linear Model", "color": "red", "subplot": 132},
            {"x": x_fig8, "y": y_fig8, "z": z_fig8, "title": "Quadrotor Figure-8 Trajectory", "color": "green", "subplot": 133}
        ]

        # Loop through the trajectory data and create each subplot
        for traj in trajectories:
            ax = fig.add_subplot(traj["subplot"], projection='3d')
            ax.plot(traj["x"], traj["y"], traj["z"], label=traj["title"], color=traj["color"])
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_zlabel("Z Position")
            ax.set_title(traj["title"])
            ax.legend()

        plt.tight_layout()
        plt.show()

    
if __name__ == "__main__":
    initial_state = [0,0,0,0,0,0,0,0,0,0,0,0] 
    target_state = [0, 30, 0, 30, 0, 30, 0, 0, 0, 0, 0, 0] 
    time_span = [0, 10]
    
    

    #usage
    controller = LQRController()
    sol_lqr_non_linear, sol_lqr_linear = controller.solve_lqr_models(initial_state, target_state, time_span)
    
    # Generate time and waypoints for figure-8 trajectory
    t_eval, waypoints = generate_waypoints_figure_8(x_amplitude, y_amplitude, omega, z0, steps)
    
    # Generate figure-8 trajectory for linear model
    x_fig8, y_fig8, z_fig8 = [], [], []
    for i in range(len(t_eval)-1):
            current_state = waypoints[i]
            target_state = waypoints[i+1]
        
            # Set up time interval for integration (e.g., from t[i] to t[i+1])
            t_interval = (t_eval[i], t_eval[i+1])
        
            # Run solve_ivp over this interval
            sol_figure_8 = solve_ivp(controller.lqr_linear, t_interval, current_state, args=(A, B, target_state))
            
            # Collect the solution points for x, y, z
            x_fig8.extend(sol_figure_8.y[1])  # Assuming x is at index 1
            y_fig8.extend(sol_figure_8.y[3])  # Assuming y is at index 3
            z_fig8.extend(sol_figure_8.y[5])  # Assuming z is at index 5

    
    controller.plot_trajectories(sol_lqr_non_linear, sol_lqr_linear, x_fig8, y_fig8, z_fig8)
    
 
    
    
    