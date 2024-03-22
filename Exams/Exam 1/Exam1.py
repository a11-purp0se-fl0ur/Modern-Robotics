"""
Description: ME:4140 Exam 1
Name: Mia Scoblic
Date: 3/6/2024
"""

# Packages and Rounding ------------------------------------------------------------------------------------------------
from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)
# ----------------------------------------------------------------------------------------------------------------------

# NOTE: I will be importing my functions as seen above. I may be pasting the applicable functions below to allow you to
#       view my functions without downloading a separate functions file. (I'm not sure how to share the functions if you
#       don't have the "Functions" folder as I do. I'll attach it to ICON either way.

# Pasted Functions -----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# EXAM BEGIN -----------------------------------------------------------------------------------------------------------

# Problem 11 ------------------------------------------------------------------------------------------------------------
print("\nProblem 11:\n")

# Given:
f1_b = np.array([-50, 0, -981])
f2_c = np.array([100, 0, -100])

# Find:
# Wrench in S (Fs)

# Calculate r vectors
r1_b = np.array([0,0,0])
r2_c = np.array([0,0,0])

F1b = Wrench(f1_b, r1_b)
print("F1b:\n", F1b)

F2c = Wrench(f2_c, r2_c)
print("F2c:\n", F2c)

# Move these wrenches to the space frame using adjoints
# Fs1 = [Ad_Tbs].T @ Fb1
# Fs2 = [Ad_Tcs].T @ Fc2
# Fs = Fs1 + Fs 2

# Finding Tbs
# Need: Rbs and pbs
Rbs = np.eye(3)
print("Rbs:\n", Rbs)
pbs = np.array([-240-1200-350, 0, -1109-675])
Tbs = constructT(Rbs, pbs)
print("Tbs:\n", Tbs)

# Finding Tcs
# Need: Rcs and pcs
Rcs = rotCombine(np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, 0, 1]))
print("Rcs:\n", Rcs)
pcs = np.array([0, -350, -1150-675])
Tcs = constructT(Rcs, pcs)
print("Tcs:\n", Tcs)

Ad_Tbs = adjoint(Tbs)

Ad_Tcs = adjoint(Tcs)

Fs1 = Ad_Tbs.T @ F1b
Fs2 = Ad_Tcs.T @ F2c

Fs = Fs1 + Fs2
print('Answer to 11:\n', Fs)
# ----------------------------------------------------------------------------------------------------------------------

# Problem 12 ------------------------------------------------------------------------------------------------------------
print("\nProblem 12:\n")
# Given:
Rbs = np.array([[0.470, -0.574, 0.671], [0.812, 0.579, -0.073], [-0.347, 0.579, 0.738]])
expCoord = np.array([0.5, 0.5, 0.5]) # In body frame

# Find:
# New Orientation Rsb'
Rbs_prime = expCoord_to_R(expCoord)

Rsb_prime = np.transpose(Rbs_prime)
print(Rsb_prime)
print('Answer to 12:\n', Rsb_prime)
print('\nSorry, no time to add any more print statements for problem 11.')
# ----------------------------------------------------------------------------------------------------------------------