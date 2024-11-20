import numpy as np
import matplotlib.pyplot as plt
from Constraints import *  # Assuming the necessary matrices like A, B, C, K, etc. are imported

class QuadrotorSimulation:
    def __init__(self):
        self.Ts = 0.01  # Sampling time
        self.T = 10  # Total simulation time
        self.n = 12  # Number of state variables
        self.ref = np.array([1, 1, 1, 0])  # Reference for x, y, z, yaw
        self.time = np.arange(0, self.T * self.Ts, self.Ts)  # Time steps
        self.Ns = len(self.time)  # Number of steps

    def non_linear_quadrotor(self, current_state, control):
        u, px, v, py, w, pz, p, phi, q, theta, r, psi = current_state

        # Adding gravity effect to the control for z-axis thrust
        control[0] += M * g

        # Thrust force
        F = control[0]

        # Linear velocity dynamics (u_dot, v_dot, w_dot)
        u_dot = r * v - q * w - g * np.sin(theta)
        v_dot = p * w - r * u + g * np.cos(theta) * np.sin(phi)
        w_dot = q * u - p * v + g * np.cos(theta) * np.cos(phi) - F / M

        # Angular velocity dynamics (p_dot, q_dot, r_dot)
        p_dot = (Jy - Jz) / Jx * q * r + control[1] / Jx  # control[1] corresponds to torque τ_φ
        q_dot = (Jz - Jx) / Jy * p * r + control[2] / Jy  # control[2] corresponds to torque τ_θ
        r_dot = (Jx - Jy) / Jz * p * q + control[3] / Jz  # control[3] corresponds to torque τ_ψ

        # Return the derivatives
        return np.array([u_dot, u, v_dot, v, w_dot, w, p_dot, p, q_dot, q, r_dot, r])

    def linear_quadrotor(self, current_state, A, B, control):
        """Computes the linear dynamics of the drone system."""
        return A @ current_state + B @ control

    def simulate_euler(self, model_type="linear", A=A, B=B):
        """Simulates the quadrotor dynamics."""
        x0 = np.zeros(self.n)  # Initial state
        xc = np.zeros(4)  # Integral state
        x_total = np.zeros((self.Ns, self.n))  # Store all states over time

        for j in range(1, self.Ns):
            # Control law
            cu = -K @ x0 + Kc @ xc  # Control input
            if model_type == "linear":
                x0 = x0 + self.linear_quadrotor(x0, A, B, cu) * self.Ts
            elif model_type == "nonlinear":
                x0 = x0 + self.non_linear_quadrotor(x0, cu) * self.Ts

            x_total[j, :] = x0  # Store state
            e = np.array([self.ref[0] - x0[1], self.ref[1] - x0[3], self.ref[2] - x0[5], self.ref[3] - x0[11]])
            xc = xc + e * self.Ts  # Update integral error

        return self.time, x_total

    def plot_states(self, time, states, title_prefix=""):
        """Plots the x, y, z positions over time."""
        plt.figure(figsize=(12, 6))

        # x-position
        plt.subplot(3, 1, 1)
        plt.plot(time, states[:, 1], label='x-position (px)', color='b')
        plt.xlabel('Time (s)')
        plt.ylabel('x (m)')
        plt.title(f'{title_prefix} x-Position over Time')
        plt.grid()
        plt.legend()

        # y-position
        plt.subplot(3, 1, 2)
        plt.plot(time, states[:, 3], label='y-position (py)', color='g')
        plt.xlabel('Time (s)')
        plt.ylabel('y (m)')
        plt.title(f'{title_prefix} y-Position over Time')
        plt.grid()
        plt.legend()

        # z-position
        plt.subplot(3, 1, 3)
        plt.plot(time, states[:, 5], label='z-position (pz)', color='r')
        plt.xlabel('Time (s)')
        plt.ylabel('z (m)')
        plt.title(f'{title_prefix} z-Position over Time')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

# Example usage
quadrotor = QuadrotorSimulation()

# Simulate linear model
time_linear, states_linear = quadrotor.simulate_euler(model_type="linear", A=A, B=B)
quadrotor.plot_states(time_linear, states_linear, title_prefix="Linear Model")

# Simulate non-linear model
time_nonlinear, states_nonlinear = quadrotor.simulate_euler(model_type="nonlinear")
quadrotor.plot_states(time_nonlinear, states_nonlinear, title_prefix="Non-linear Model")
