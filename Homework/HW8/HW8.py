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

S = np.column_stack((s1, s2, s3))

# Given Thetas
theta1 = np.radians(30)
theta2 = np.radians(30)
theta3 = np.radians(30)

theta = np.column_stack((theta1, theta2, theta3))

T = PoE_Space(theta, M, S)
print('\nT:\n', T)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:')

# Given
L1 = L2 = L3 = 100
b = 50
theta1 = np.radians(45)
theta2 = np.radians(35)
theta3 = np.radians(-45)
theta = np.column_stack([theta1, theta2, theta3])

# Calculate M
R = np.eye(3)
p = np.array([0, L2+L3, L1+b])
M = constructT(R, p)

# Calculate Screws
h = 0  # All joints are R
sHat1 = np.array([0, 0, 1])
sHat2 = np.array([1, 0, 0])
sHat3 = np.array([-1, 0, 0])
q1 = np.array([0, 0, b])
q2 = np.array([0, 0, b+L1])
q3 = np.array([0, L2, b+L1])

s1 = parametersToScrew(sHat1, q1, h)
s2 = parametersToScrew(sHat2, q2, h)
s3 = parametersToScrew(sHat3, q3, h)

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

f = np.array([5, -1, 3])
m = np.array([3, 1, 2])

