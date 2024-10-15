import numpy as np
import matplotlib.pyplot as plt


def figure_8(A, B, omega, z0, t):
    
    x = A * np.sin(omega * t)                # x(t) = A * sin(ωt)
    y = B * np.sin(2 * omega * t)            # y(t) = B * sin(2ωt)
    z = np.full_like(t, z0)                   # z(t) = z0 (constant altitude)
    return x, y, z

def plot_figure_8(A, B, omega, z0):
    t = np.linspace(0, 20, 1000)              # Time from 0 to 20 seconds, 1000 samples
    x, y, z = figure_8(A, B, omega, z0, t)

  
    fig = plt.figure(figsize=(20, 16), facecolor='lightgray')

    # 3D Plot
    ax_3d = fig.add_subplot(221, projection='3d')
    ax_3d.plot(x, y, z, color='blue', linewidth=2, label='Figure-8 Trajectory')
    ax_3d.set_title('3D Trajectory', fontsize=12, fontweight='bold')
    ax_3d.set_xlabel('X Position (m)', fontsize=10)
    ax_3d.set_ylabel('Y Position (m)', fontsize=10)
    ax_3d.set_zlabel('Z Position (m)', fontsize=10)
    ax_3d.grid(True, linestyle='-', alpha=0.7)
    ax_3d.legend()

    # 2D Subplots for X, Y, Z
    colors = ['red', 'green', 'blue']
    titles = ['X Position over Time', 'Y Position over Time', 'Z Position over Time']
    y_labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']

    for i, (data, title, ylabel, color) in enumerate(zip([x, y, z], titles, y_labels, colors)):
        ax = fig.add_subplot(2, 2, i + 2)
        ax.plot(t, data, color=color, linewidth=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='-', alpha=0.7)

    plt.suptitle('Figure-8 Trajectory Visualization', 
                 fontsize=14, fontweight='bold', fontstyle='italic', color='darkblue')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.1, hspace=0.4, wspace=0.4)
    plt.show()


plot_figure_8(A=2.0, B=1.0, omega=0.5, z0=1.0)
