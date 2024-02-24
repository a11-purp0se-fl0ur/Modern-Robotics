"""
Description: ME:4140 Homework 5
Name: Mia Scoblic
Date: 2024-02-21
"""
import numpy as np

from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')
# Given:
q1 = np.array([1, 3, 6])
s1 = np.array([3, 2, 1])
s1Hat = normalize(s1)
h1 = 4

# Find: Screw from parameters
S = parametersToScrew(s1Hat, q1, h1)
print("S:\n", S)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:')
# Given:
Tbc = np.array([[1, 0, 0, 5],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])

# Find Tab
#   Plan: Invert Tbc, and multiply by Tac
#   Tac @ Tcb = Tab

# Invert Tbc
Tcb = np.linalg.inv(Tbc)

# Find Tac
#   Find Rac
xac = np.array([0, 1, 0])
yac = np.array([1, 0, 0])
zac = np.array([0, 0, -1])
Rac = rotCombine(xac, yac, zac)
#   Find pac
pac = np.array([-1, 1, 2])
#   Construct Tac
Tac = constructT(Rac, pac)

# Calculate Tab
Tab = Tac @ Tcb
print("Tab:\n", Tab)


# Problem 3 ------------------------------------------------------------------------------------------------------------
print('\nProblem 3:')
# Given:
Vb = np.array([1, 2, 2, 0, 0, 0])

# Find: Vc
#   Need the adjunct of Tcb
Adj_Tcb = adjoint(Tcb)

# Calculate Va
Vc = Adj_Tcb @ Vb
print("Va:\n", Vc)

# Problem 4 ------------------------------------------------------------------------------------------------------------
print('\nProblem 4:')
# Given:
# Pure Rotation
q4 = np.array([-2, 1, 0])
theta4 = 7
h4 = 0
sHat4 = np.array([0, 0, 1])

# Find: Va
#   Find Screw S
S4 = parametersToScrew(sHat4, q4, h4)

#   Va is this screw S multiplied by omega
Va4 = S4 * theta4
print("Va:\n", Va4)

# Problem 5 ------------------------------------------------------------------------------------------------------------
print('\nProblem 5:')
# Given:
q5 = np.array([0, 3, 0])
h5 = 5

# Find: Sc
#   Calculate sHat
sHat5 = np.array([0, np.cos(np.radians(45)), np.sin(np.radians(45))])

#   Calculate Sc
Sc = parametersToScrew(sHat5, q5, h5)
print("Sc:\n", Sc)

# Problem 6 ------------------------------------------------------------------------------------------------------------
print('\nProblem 6:')
# Given:
thetaDot6 = 2

# Find: Vc
Vc6 = Sc * thetaDot6
print("Vc:\n", Vc6)