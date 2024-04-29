import numpy as np

from Functions.Mia_Functions import *
import modern_robotics as mr
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:\n')

# Given:
theta_rad = np.array([np.pi/4, -np.pi/4, -np.pi/2, 0])
L1 = L2 = L3 = L4 = 1.5

# Find: The Jacobian from the previous given thetas

# Home Position
R = np.eye(3)
p = np.array([L1+L2+L3+L4, 0, 0])
M = constructT(R, p)

# Screws
h = 0
sHat1 = sHat2 = sHat3 = sHat4 = np.array([0, 0, 1])
q1 = np.array([0, 0, 0])
q2 = np.array([L1, 0, 0])
q3 = np.array([L1+L2, 0, 0])
q4 = np.array([L1+L2+L3, 0, 0])
s1 = parametersToScrew(sHat1, q1, h)
s2 = parametersToScrew(sHat2, q2, h)
s3 = parametersToScrew(sHat3, q3, h)
s4 = parametersToScrew(sHat4, q4, h)
Ss = np.column_stack([s1, s2, s3, s4])

J = SpaceJacobian(Ss, theta_rad)
print(J)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:\n')

# Given:
J = np.array([[1, 3],
              [2, 4],
              [3, 3]])

# Find: Determine the Inverse
Jinv = pseudoInv(J)
Jinv2 = np.linalg.pinv(J)
print(Jinv)
print(Jinv2)

