import math
from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Given
L1 = L2 = L3 = 0.1
b = 0.05

# Define Home Position
R = np.eye(3)
p = np.array([0, L1+L2, b+L1])
M = constructT(R,p)

print('\nM:\n', M)

# Define joint angles
theta1 = np.pi/3
theta2 = np.pi/2
theta3 = np.pi/3

theta = np.column_stack((theta1,theta2,theta3))

print('\ntheta:\n', theta)

# Define screws
h = 0
q1 = np.array([0, 0, b])
q2 = np.array([0, 0, b+L1])
q3 = np.array([0, L2, b+L1])
sHat1 = np.array([0, 0, 1])
sHat2 = np.array([1, 0, 0])
sHat3 = np.array([1, 0, 0])

S1 = parametersToScrew(sHat1, q1, h)
S2 = parametersToScrew(sHat2, q2, h)
S3 = parametersToScrew(sHat3, q3, h)

S = np.column_stack((S1, S2, S3))

print('\nS:\n', S)

# On to Jacobian
Js1 = S1
print('\nJs1:\n', Js1)

# Js2
exp1 = expCoord_to_T(S1, theta1)
exp1_Adj = adjoint(exp1)
Js2 = exp1_Adj @ S2
print('\nJs2:\n', Js2)

# Js3
exp2 = expCoord_to_T(S2, theta2)
combine = exp1 @ exp2
combine_Adj = adjoint(combine)
Js3 = combine_Adj @ S3
print('\nJs3:\n', Js3)

# Construct Jacobian
J = np.column_stack((Js1, Js2, Js3))
print('\nJacobian:\n', J)