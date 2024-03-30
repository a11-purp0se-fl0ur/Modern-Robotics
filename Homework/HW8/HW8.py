import numpy as np

from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')

M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 200], [0, 0, 0, 1]])

s1 = np.array([0, 1, 0, 0, 0, 0])
s2 = np.array([0, 1, 0, 0, -100, 0])
s3 = np.array([0, 1, 0, 0, -150, 0])

S = np.column_stack((s1, s2, s3))

theta1 = np.radians(30)
theta2 = np.radians(30)
theta3  np.radians(30)