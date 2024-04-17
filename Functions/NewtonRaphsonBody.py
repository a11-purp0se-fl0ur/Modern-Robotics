import modern_robotics as mr
from Functions.Mia_Functions import *

# Decription: Finds the joint angles that leads to desired configuration. Provide space frame values and it will conver to body if applicable.
def NewtonRaphson(S, M, Tsd, theta0, eomg, ev, frame):
    if frame == 'space':

    if frame == 'body':


Tsd = np.array([ [-0.5, -0.866, 0, 0.366],
                 [0.866, -0.5,   0, 1.366],
                 [0,     0,     1, 0],
                 [0,     0,     0, 1] ])

Rsb = np.eye(3)
p = np.array([2, 0, 0])
M = constructT(Rsb,p)

# Robot screws in the space frame
S = np.array([ [0, 0],
               [0, 0],
               [1, 1],
               [0, 0],
               [0, -1],
               [0, 0] ])

# Robot screws in the body frame
B = adjoint(np.linalg.inv(M)) @ S

eomg = 1e-3 # rotational error, rad
ev = 1e-3 # translational error, m

theta_deg0 = np.array([10,10])
theta_rad0 = np.deg2rad(theta_deg0)

[thetalist,success] = mr.IKinBody(B,M,Tsd,theta_rad0,eomg,ev)

thetalistdeg = np.rad2deg(thetalist)

print(thetalistdeg)
