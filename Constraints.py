from scipy.io import matlab

# Drone parameters
mq = 0.6  # mass of the quadrotor (Kg)
L = 0.2159  # arm length (m)
g = 9.81  # acceleration due to gravity (m/s^2)
ms = 0.410  # mass of the central sphere (Kg)
R = 0.0503513  # radius of the sphere (m)
mprop = 0.00311  # mass of the propeller (Kg)
mm = 0.036 + mprop  # mass of the motor + propeller (Kg)

# Moments of inertia
Jx = (2 * ms * R**2) / 5 + 2 * L**2 * mm
Jy = (2 * ms * R**2) / 5 + 2 * L**2 * mm
Jz = (2 * ms * R**2) / 5 + 4 * L**2 * mm

#import K
data= matlab.loadmat('Operations_Research/K.mat')
K = data['K'] 


#import Kc
data1=matlab.loadmat('Operations_Research/Kc.mat')
Kc= data1['Kc']


#slide 35 
A = [
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
]

#slide 35
B = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-1/mq, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1/Jx, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1/Jy, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1/Jz],
    [0, 0, 0, 0],   
]
C = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]
#ref= [x, y,z, p(0)]
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
