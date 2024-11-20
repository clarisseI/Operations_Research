import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Constraints import *  # Assuming the necessary matrices like A, B, C, K, etc. are imported
from Figure_8_Trajectory import generate_waypoints_figure_8

class IntegralControl:
    def __init__(self):
        pass
    
    def integral_control_linear(self, time, current_state, Ac, Bc, target_state):
        current_state = np.array(current_state)
        target_state = np.array(target_state)

        # Compute control with correct indexing
        control = current_state[[1, 3, 5, 11]] - target_state[[1, 3, 5, 11]]
        
        dx = Ac @ current_state + Bc @ control
        return dx
    
    
    
    
    def integral_control_non_linear(self, time, current_state, target_state):
        # the current state
        u, px, v, py, w, pz, p, phi, q, theta, r, psi,t1,t2,t3,t4 = current_state

        current_state = np.array(current_state)
        target_state = np.array(target_state)
        
        error= current_state[:12] - target_state[:12]
        # Control law (feedback control based on error)
        #combined_error = np.hstack((error, error))  # Since integral_error and error are the same

        # Compute the control input
        control = -K @ error +Kc @ current_state

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
        dx = np.array([u_dot,u,v_dot,v,w_dot,w, p_dot,p, q_dot,q,r_dot,r,t1,t2,t3,t4])

        state = []
        for el in dx:
            el = float(el)
            state.append(el)

        return state


    # Integral control LQR function to compute augmented system dynamics
    def solve_integral_control(self, initial_state, target_state, time_span):
        
        # Augmented initial and target states (with integral states initialized to zero)
        aug_initial_state = np.concatenate([initial_state, np.zeros(q)])
        aug_target_state = np.concatenate([target_state, np.zeros(q)])

        # Solve integral control dynamics using solve_ivp
        sol_integral_control = solve_ivp(self.integral_control_linear, time_span, aug_initial_state,
                            args=(Acl, Bcl, aug_target_state), dense_output=True)
        return sol_integral_control



    def plot_results(self, sol_integral_control, x_fig8, y_fig8, z_fig8):

         # Extract x, y, z positions from solv_integral_control
        x_inc, y_inc, z_inc = sol_integral_control.y[1], sol_integral_control.y[3], sol_integral_control.y[5]

        # Create a figure 
        fig = plt.figure(figsize=(18, 6))
    
        # Data and labels for each subplot
        trajectories = [
            {"x": x_inc, "y": y_inc, "z": z_inc, "title": "Integral Control linear Model", "color": "blue", "subplot": 121},
            {"x": x_fig8, "y": y_fig8, "z": z_fig8, "title": "Quadrotor Figure-8 Trajectory", "color": "green", "subplot": 122}
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


def main():
    
    initial_state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
    target_state = [0, 30, 0, 30, 0, 30, 0, 0, 0, 0, 0, 0] 
    time_span = [0, 10]
    
    
    #usage
    controller = IntegralControl()
    solv_integral_control= controller.solve_integral_control(initial_state, target_state, time_span)
    
    """
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
            sol_figure_8 = solve_ivp(controller.integral_control_linear, t_interval, current_state, args=(A, B, target_state, K))
            
            # Collect the solution points for x, y, z
            x_fig8.extend(sol_figure_8.y[1])  # Assuming x is at index 1
            y_fig8.extend(sol_figure_8.y[3])  # Assuming y is at index 3
            z_fig8.extend(sol_figure_8.y[5])  # Assuming z is at index 5"""

    # Plot the results: LQR control and Figure-8 trajectory combined
    controller.plot_results(solv_integral_control, x_fig8=None, y_fig8=None, z_fig8=None)

if __name__ == "__main__":
    main()


'''
Xc:integral with 4 zeros
'''