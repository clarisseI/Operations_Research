import numpy as np
from Integral_Control_Class import IntegralControl
from Constraints import *

def euler_method_linear_integral(target_state, initial_state, total_time=10, time_step=0.01):
    """
    Simulates a linear system using Euler's method.

    Args:
        target_state (ndarray): Target state vector.
        initial_state (ndarray): Initial state vector.
        total_time (float): Total simulation time.
        time_step (float): Time step for the simulation.

    Returns:
        state_trajectory (ndarray): State vector at each time step.
        time_steps (ndarray): Time steps.
    """
    num_steps = int(total_time / time_step) + 1
    time_steps = np.linspace(0, total_time, num_steps)
    state_trajectory = np.zeros((initial_state, num_steps))
    current_state = initial_state
    state_trajectory[:, 0] = initial_state

    integral_control = IntegralControl()

    for i in range(1, num_steps):
        dx = integral_control.integral_control_linear(i, current_state, Acl, Bcl, target_state)
        current_state += dx * time_step
        state_trajectory[:, i] = current_state

   ''''
   
   Ts=0.01
   T=10
   tt= np.arange( T* Ts, Ts)
   Ns= tt *size
   n=12
   x0= np.zeros(n)
   xc=np.zeros(4)
   ref= np.zeros[4)
   ref[0]=1
   ref[1]=1
   ref[2]=1
   ref[3]=0
   x_total= np.zeros((Ns, n))
   
    for j in range(1, Ns):
     cu= -k @x0 + kc @xc
     x0= x0+ quadrator(xo, cu)*ts
     x[total: j]= xo
     e= np.array(ref[0]- xo[1], ref[1]-xo[3], ref[2]-xo[5], ref[2]-ref[11])
     xc= xc+e +ts
     
   
   
   Keep A and B instead of Bc and Ac
   '''

   