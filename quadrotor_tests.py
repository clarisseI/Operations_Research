import numpy as np
import matplotlib.pyplot as plt
from dynamic_operation import solve_quadrotor_dynamics  

def set_labels(ax, title, xlabel, ylabel):
    #function to set title and labels for the axes
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)


def display_plot(t, x_nl, y_nl, z_nl, x_l, y_l, z_l, target_state, time_span, feedforward=None):
    # A visualization with a 3D plot and three 2D plots for linear and non-linear control.
    
    fig = plt.figure(figsize=(20, 16), facecolor='lightgray')

    # 3D Plot
    ax_3d = fig.add_subplot(221, projection='3d')
    ax_3d.plot(x_nl, y_nl, z_nl, color='blue', linewidth=2, label='Non-linear')
    ax_3d.plot(x_l, y_l, z_l, color='red', linewidth=2, linestyle='--', label='Linear')

    # Set titles and labels for the 3D plot
    set_labels(ax_3d, '3D Trajectory', 'X Position', 'Y Position')
    ax_3d.set_zlabel('Z Position', fontsize=10)
    ax_3d.grid(True, linestyle='-', alpha=0.7)
    ax_3d.legend()

    # Create 2D Subplots for X, Y, Z with distinct colors
    colors = ['red', 'green', 'blue']
    titles = ['X Position over Time', 'Y Position over Time', 'Z Position over Time']
    y_labels = ['X Position', 'Y Position', 'Z Position']

    for i, (data_nl, data_l, title, ylabel, color) in enumerate(zip(
            [x_nl, y_nl, z_nl], [x_l, y_l, z_l],
            titles, y_labels, colors)):
        
        ax = fig.add_subplot(2, 2, i + 2)
        ax.plot(t, data_nl, color=color, linewidth=2, label='Non-linear')
        ax.plot(t, data_l, color=color, linewidth=2, linestyle='--', label='Linear')
        set_labels(ax, title, 'Time (s)', ylabel)
        ax.grid(True, linestyle='-', alpha=0.7)
        ax.legend(fontsize=8)

    plt.suptitle(f'Quadrotor Dynamics: Non-linear vs Linear Control\n'
                 f'Target: [{target_state[1]}, {target_state[3]}, {target_state[5]}] | '
                 f'Time Span: {time_span}', 
                 fontsize=10, fontweight='bold', fontstyle='italic')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.1, hspace=0.4, wspace=0.4)
    plt.show()

if __name__ == "__main__":
    initial_state = [0,0,0,0,0,0,0,0,0,0,0,0] 
    target_state = [0, 100, 0, 100, 0, 100, 0, 0, 0, 0, 0, 0] 
    time_span = [0, 5]
    time_eval = np.linspace(0, 5, 250) 
   
 
    sol_non_linear, sol_linear= solve_quadrotor_dynamics(initial_state, target_state, time_span, time_eval)


    x_nl, y_nl, z_nl = sol_non_linear.y[1], sol_non_linear.y[3], sol_non_linear.y[5]  # Non-linear positions
    x_l, y_l, z_l = sol_linear.y[1], sol_linear.y[3], sol_linear.y[5]  # Linear positions
    t = sol_linear.t  # Time vector

    # Call the display_plot function to generate the graphs
    display_plot(t, x_nl, y_nl, z_nl, x_l, y_l, z_l, target_state, time_span)
