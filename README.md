# Operations_Research

Initial state =[0]\*12
current state= u px v py w pz thetadot, theta, phidot, phi, phydot, phy

Target state= [0, pxtarget,
0, pytarget,
0, pztarget,
0,0
0,0
0,pttarget=0]

Solution= solve_ip( time, x(initial state))

Control= K.(currentstate- targetstate)
control[0]= control[0]+ mg
mg=
g= 9.81

Def non_linear(time, current state, target_state, K){
build u v, w, phi theta, psi from current_State

    //compute the control
    control= -K( current-target)
    control[0]= control[0]+ mg
    f= control[0]
    du
    dv
    dw
    dp
    dq
    dv

    i/jx= control[1]
    1/jy= control[2]
    1/jz= control[3]

    dx= [du, u, dv, v, dw, w, dp, p, dq, q, dv, v]

    return dx

}

def linear( time, current_state, A, B, target_state){
control= -K(current- target)
dx= Ax +Bcontrol
return dx
}

Reference:

- loading .mat
  https://stackoverflow.com/questions/6780080/open-matlab-file-mat-with-module-pickle-in-python
#when we use we need to add mg for non_linear instead here we don't do it
# for linear when we simulate it and get out of the gravity