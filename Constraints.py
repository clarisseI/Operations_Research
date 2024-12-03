import scipy.io
import numpy as np
import control as ctrl
# Drone parameters
M = 0.6  # mass of the quadrotor (Kg)
L = 0.2159  # arm length (m)
g = 9.81  # acceleration due to gravity (m/s^2)
m = 0.410  # mass of the central sphere (Kg)
R = 0.0503513  # radius of the sphere (m)
m_prop = 0.00311  # mass of the propeller (Kg)
m_m = 0.036 + m_prop  # mass of the motor + propeller (Kg)

# Moments of inertia
Jx = (2 * m * R) / 5 + 2 * L**2 * m_m
Jy = (2 * m * R) / 5 + 2 * L**2 * m_m
Jz = (2 * m * R) / 5 + 4 * L**2 * m_m



# Import K and Extract the K matrix
K = scipy.io.loadmat('K.mat')['K']
#import Kc and Extract the K matrix
Kc = scipy.io.loadmat('Kc.mat')['Kc']


#Linearlized Model in Hovering Mode

# Define the A matrix
A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0,-g, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
])

# Define the B matrix
B =np.array( [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-1/M, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1/Jx, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1/Jy, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1/Jz],
    [0, 0, 0, 0],   
])

# Define the C matrix
C =np.array( [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])



ref=[1,1,1,1]

Ac= [
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],

    ]
Bc= [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],

    ]

n = A.shape[0]
p = B.shape[1]
q = C.shape[0]

D = np.zeros((q, p))

# LQR Controller
Ac = np.zeros((q, q))
Bc = np.eye(q)

hat_A = np.block([[A,np.zeros((n,q))],
                    [Bc @ C, Ac]])

hat_B = np.vstack((B, np.zeros((q, p))))

hat_C = np.hstack((C, np.zeros((q, q))))
hat_D = np.zeros((q, p))

# Closed-loop system matrices
Acl = np.block([[A - B @ K, B @ Kc],
                        [-Bc @ C, Ac]])
Bcl = np.vstack([np.zeros((n, p)), Bc])


Ccl= hat_C
Dcl= np.zeros((q,p))

# Continuous to discrete conversion
ts1 = 0.1 # Sampling time
sys = ctrl.ss(Acl, Bcl, Ccl, Dcl)
sysd = ctrl.c2d(sys, ts1)
Ad = sysd.A
Bd = sysd.B

# Define the constraint matrix S
S = np.block([[-K, Kc],
              [K, -Kc]])

K_hat= np.block([K, -Kc])  # Combine K and -Kc into a single matrix

s = np.array([6, 0.005, 0.005, 0.005, 6, 0.005, 0.005, 0.005]).T
# Figure 8 parameters
x_amplitude = 2         
y_amplitude = 1         
omega = 0.5             
z0 = 1                  
steps = 1000 