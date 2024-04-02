import numpy as np

from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')

# Given M
M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 200], [0, 0, 0, 1]])

# Given Screws
s1 = np.array([0, 1, 0, 0, 0, 0])
s2 = np.array([0, 1, 0, 0, -100, 0])
s3 = np.array([0, 1, 0, 0, -150, 0])

# Screw Array
S = np.column_stack((s1, s2, s3))

# Given Thetas
theta1 = np.radians(30)
theta2 = np.radians(30)
theta3 = np.radians(30)

# Theta Array
theta = np.column_stack((theta1, theta2, theta3))

# Construct T
T = PoE_Space(theta, M, S)
print('\nT:\n', T)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:')

# Given
L1 = L2 = L3 = 0.1
b = 0.05
theta1 = np.radians(45)
theta2 = np.radians(35)
theta3 = np.radians(-45)

# Theta Array
theta = np.column_stack([theta1, theta2, theta3])

# Calculate M
R = np.eye(3)
p = np.array([0, L2+L3, L1+b])
M = constructT(R, p)

# Calculate Screws
h = 0  # All joints are R

# Screw unit axes
sHat1 = np.array([0, 0, 1])
sHat2 = np.array([1, 0, 0])
sHat3 = np.array([1, 0, 0])

# Screw distance from {s}
q1 = np.array([0, 0, b])
q2 = np.array([0, 0, b+L1])
q3 = np.array([0, L2, b+L1])

# Calculate screws
s1 = parametersToScrew(sHat1, q1, h)
s2 = parametersToScrew(sHat2, q2, h)
s3 = parametersToScrew(sHat3, q3, h)

# Screw array
S = np.column_stack((s1, s2, s3))

# Calculate T
T = PoE_Space(theta, M, S)
print('\nT:\n', T)

# Problem 3 ------------------------------------------------------------------------------------------------------------
print('\nProblem 3:')

# Given
theta1 = np.pi/3
theta2 = np.pi/2
theta3 = np.pi/3
f = np.array([5, -1, 3])  # Applied force at tip
m = np.array([3, 1, 2])   # Applied moment at tip

# Position vector from {s} to {ee}
Fe = np.array([f[0], f[1], f[2], m[0], m[1], m[2]])

# Construct Tes
Res = np.eye(3)
pes = np.array([0, -L2-L3, -L1-b])
Tes = constructT(Res, pes)

# Adjoint Tes
adj_Tes = adjoint(Tes)

# Move wrench to {s}
Fs = adj_Tes.T @ Fe
print('Wrench in space frame:\n', Fs)

# Calculate torque
torque = S.T @ Fs
print('\nTorque:\n', torque)

# Problem 4 ------------------------------------------------------------------------------------------------------------
print('\nProblem 4:')

# Given Dimensions
A = 3.5
B = 6.75
C = 11.50
D = 0.41
E = 12.0
F = 2.4

# Given force and moment
fb = np.array([-50, 0, -981])
mb = np.array([0, 0, 0])

# Position vector from {s} to {b}
pbs = np.array([A+E+F, 0, B+C-D])

# Transformation from {s} to {b}
Rbs = np.eye(3)
Tbs = constructT(Rbs, pbs)

# Wrench in {s}
Fs = Wrench(fb, pbs)

# Initialize Thetas
theta = np.column_stack([0, 0, 0, 0, 0, 0])

# Screw parameters
h = 0
wHat1 = np.array([0, 0, 1])
wHat2 = np.array([0, 1, 0])
wHat3 = np.array([0, 1, 0])
What4 = np.array([1, 0, 0])
wHat5 = np.array([0, 1, 0])
wHat6 = np.array([1, 0, 0])



