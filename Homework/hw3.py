"""
Description: ME:4140 Homework 2
Name: Mia Scoblic
Date: 2024-02-03
"""

import numpy as np
from My_Functions.Rotation_Matrices_Functions import *

# Problem 1 part a
print('\nProblem 1, Part a:')
x_a = np.array([0, 0, 1])
y_a = np.array([-1, 0, 0])

Rsa = s_a(x_a, y_a)
print('\nRsa:\n', Rsa)

# Problem 1 part b
print('\nProblem 1, Part b:')
x_b = np.array([1, 0, 0])
y_b = np.array([0, 0, -1])

Rsb = s_a(x_b, y_b)
print('\nRsb:\n', Rsb)

# Problem 1 part c
print('\nProblem 1, Part c:')
Rbs = a_s(x_b,y_b)
print('\nRbs:\n', Rbs)

# Problem 1 part d
print('\nProblem 1, Part d:')
Ras = a_s(x_a, y_a)

Rab = a_b(Ras, Rsa)
print('\nRab:\n', Rab)

# Problem 1 part e

print('\nProblem 1, Part e:')
p_a = np.array([1, -1, 0])
p_s = point(p_a, Rsb)
print('\np_s:\n', p_s)

# Problem 6
print('\nProblem 6:')
Rsa = rot