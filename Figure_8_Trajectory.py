import numpy as np

def generate_waypoints_figure_8(x_amplitude, y_amplitude, omega, z0, steps):
    """
    Generates waypoints for the desired 3D trajectory (a figure-8 in the x-y plane).
    """
    t = np.linspace(0, 4 * np.pi / omega, steps)
    x = x_amplitude * np.sin(omega * t)
    y = y_amplitude * np.sin(2 * omega * t)
    z = np.full_like(t, z0)

    waypoints = []
    for i in range(steps):
        waypoints.append([0, x[i], 0, y[i], 0, z[i], 0, 0, 0, 0, 0, 0])

    return t, waypoints

# drone is close to the current waypoints before generating the next one




##### Short distance and long distance to compare





