from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')

# Given
Tsd = np.array([[0.707, -0.696, -0.123, -127.5], [0.707, 0.696, 0.123, 127.5], [0, -0.174, 0.985, 190], [0, 0, 0, 1]])
L1 = L2 = L3 = 100
b = 50

# Find the joint angles to achieve Tsd

# Since we are given T instead of a coordinate location of {ee}, we will deconstruct T
R, p = deconstructT(Tsd)

# p is now the desired coordinates of the end effector

# desired