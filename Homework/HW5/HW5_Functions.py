"""
Description: ME:4140 Homework 5
Name: Mia Scoblic
Date: 2024-02-21
"""

from Functions.Mia_Functions import *
from Functions.Phil_Functions import *

# Problem 1 ------------------------------------------------------------------------------------------------------------
# Find: Screw axis from parameters
print('\nProblem 1:')

q1 = np.array([1, 3, 6])
s1 = np.array([3, 2, 1])
h1 = 4

S = parametersToScrew(s1, q1, h1)
print("S:\n", S)