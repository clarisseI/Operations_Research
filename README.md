# Quadrotor Simulator

## Overview

This project is a Python-based simulator for quadrotor dynamics. The simulator aims to visualize the quadrotor's motion and implement various control strategies, including Linear and Non-Linear Quadratic Regulator control methods. The project also allows for testing various trajectories, including a figure-8 pattern.

## Features

- **Mathematical Modeling**: Develop a comprehensive model of quadrotor dynamics.
- **Control Strategies**:
  - LQR with feedforward control
  - LQR with integral control
- **Trajectory Testing**: Implement various trajectories, including a figure-8 pattern.
- **Visualization**: Use Matplotlib for real-time visualization of the quadrotor's motion.

## Requirements

To run this project, you will need:

- Python 3.x
- NumPy
- Matplotlib
- scipy

You can install the required packages using pip:

```bash
pip install numpy matplotlib scipy
```

## Getting Started

### Clone the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/quadrotor-simulator.git
cd quadrotor-simulator
```

### Running the Simulator

To run the simulator, use the following command:

```bash
python simulator.py
```

### Modifying Target States

You can modify the target state values in the simulator by changing the indices in the code. The current indices for testing are 7, 9, and 11.

## Project Structure

```
quadrotor-simulator/
├── simulator.py          # Main simulation script
├── quadrotor_model.py    # Quadrotor dynamics model
├── controller.py         # Control strategy implementations
├── trajectories.py       # Trajectory definitions
└── README.md             # Project documentation
```

