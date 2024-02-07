"""
Description: ME:4140 Homework 2
Name: Mia Scoblic
Date: 2024-02-03
"""

import numpy as np
from Functions.Mia_Functions import *
from Functions.Phil_Functions import *

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')
x_a = np.array([0, 0, 1])
y_a = np.array([-1, 0, 0])
Rsa = s_a(x_a, y_a)
print('Rsa:\n', Rsa)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:')
x_b = np.array([1, 0, 0])
y_b = np.array([0, 0, -1])
Rsb = s_a(x_b, y_b)
print('Rsb:\n', Rsb)

# Problem 3 ------------------------------------------------------------------------------------------------------------
print('\nProblem 3:')
Rbs = np.transpose(Rsb)
print('Rbs:\n', Rbs)

# Problem 4 ------------------------------------------------------------------------------------------------------------
print('\nProblem 4:')
Ras = np.transpose(Rsa)
Rab = a_b(Ras, Rsb)
print('Rab:\n', Rab)

# Problem 5 ------------------------------------------------------------------------------------------------------------
print('\nProblem 5:')
Pa = np.array([3, 2, 1])
Ps = Rsa @ Pa
print('Ps:\n', Ps)

# Problem 6 ------------------------------------------------------------------------------------------------------------
print('\nProblem 6:')
Rsa_2 = Rot('x', np.pi/4, 'rad') @ Rot('z', np.pi/2, 'rad')
Z = np.round(Rsa_2, 1)
print('Rsa:\n', Z)

# Problem 7 ------------------------------------------------------------------------------------------------------------
print('\nProblem 7:')
Ras_2 = np.transpose(Z)
print('Ras:\n', Ras_2)

# Problem 8 ------------------------------------------------------------------------------------------------------------
print('\nProblem 8:')
Rsb_2 = Rot('z', 60, 'deg') @ Rot('x', 30, 'deg') @ Rot('y', 90, 'deg')
Y = np.round(Rsb_2, 1)
print('Rsb:\n', Y)

# Problem 9 ------------------------------------------------------------------------------------------------------------
print('\nProblem 9:')
Rbs_2 = np.transpose(Y)
print('Rbs:\n', Rbs_2)

# Problem 10 -----------------------------------------------------------------------------------------------------------
print('\nProblem 10:')
Rab_2 = Ras_2 @ Rsb_2
X = np.round(Rab_2, 1)
print('Rbs:\n', X)

# Problem 11 -----------------------------------------------------------------------------------------------------------
print('\nProblem 10:')
Rba_2 = np.transpose(X)
print('Rba:\n', Rba_2)

