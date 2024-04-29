import numpy as np

from Functions.Mia_Functions import *
import modern_robotics as mr
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:\n')
# Given:
L1 = 550
L2 = 300
L3 = 60
W1 = 45
theta_deg = np.array([0, 45, 0, -45, 0, -90, 0])
theta_rad = np.deg2rad(theta_deg)

# Define Home Position
R = np.eye(3)
p = np.array([0, 0, L1+L2+L3])
M = constructT(R, p)

# Define Screw in {s}
h = 0
sHat1357 = np.array([0, 0, 1])
sHat246 = np.array([0, 1, 0])
q1 = q2 = q3 = np.array([0, 0, 0])
q4 = np.array([W1, 0, L1])
q5 = q6 = q7 = np.array([0, 0, L1+L2])
s1 = parametersToScrew(sHat1357, q1, h)
s2 = parametersToScrew(sHat246, q2, h)
s3 = parametersToScrew(sHat1357, q3, h)
s4 = parametersToScrew(sHat246, q4, h)
s5 = parametersToScrew(sHat1357, q5, h)
s6 = parametersToScrew(sHat246, q6, h)
s7 = parametersToScrew(sHat1357, q7, h)
Ss = np.column_stack((s1, s2, s3, s4, s5, s6, s7))

# Convert to {ee}
Sb = adjoint(np.linalg.pinv(M)) @ Ss

# Calculate PoE
T = PoE_Body(theta_rad, M, Sb)
print('T:\n', T)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:\n')
# Given:

theta_deg = np.array([0, -90, 0, 0, 90, 0, 0])
theta_rad = np.deg2rad(theta_deg)

L1 = 350
L2 = L3 = 410
L4 = 136

# Home Position
R = np.eye(3)
p = np.array([0, 0, L1+L2+L3+L4])
M = constructT(R, p)

# Screws
h = 0
sHat246 = np.array([1, 0, 0])
sHat1357 = np.array([0, 0, 1])
q1 = q3 = q5 = q7 = np.array([0, 0, 0])
q2 = np.array([0, 0, L1])
q4 = np.array([0, 0, L1+L2])
q6 = np.array([0, 0, L1+L2+L3])

s1 = parametersToScrew(sHat1357, q1, h)
s2 = parametersToScrew(sHat246, q2, h)
s3 = parametersToScrew(sHat1357, q3, h)
s4 = parametersToScrew(sHat246, q4, h)
s5 = parametersToScrew(sHat1357, q5, h)
s6 = parametersToScrew(sHat246, q6, h)
s7 = parametersToScrew(sHat1357, q7, h)

Ss = np.column_stack([s1, s2, s3, s4, s5, s6, s7])

# Final Config
T = PoE_Space(theta_rad, M, Ss)
print('T:\n', T)

