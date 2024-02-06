"""
Description: ME:4140 Homework 1
Name: Mia Scoblic
Date: 2021-01-28
"""

import numpy as np

# Problem 1
print('\nProblem 1:')
A = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

AT = np.transpose(A)
print('\nTranspose of A is:\n', AT)

# Problem 2
print('\nProblem 2:')
B = np.array([[1, 2, 3, 4],
              [9, 2, 0, -1],
              [3, -3, 1, 2],
              [-3, 5, 6, 1]])

C = np.array([[0, 1, 0, 4],
              [9, 5, -4, 4],
              [6, 3, -8, 0],
              [3, 2, 2, 1]])

BC = B @ C
print('B X C = :\n', BC)

# Problem 3
print('\nProblem 3:')
D = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

dTrace = np.trace(D)
print('Trace of D is:\n', dTrace)

# Problem 4
print('\nProblem 4:')
E = np.array([[0, 1, 0, 4],
              [9, 5, -4, 4],
              [6, 3, -8, 0],
              [3, 2, 2, 1]])

eTrace = np.trace(E)
print('Trace of E is:\n', eTrace)

# Problem 5
print('\nProblem 5:')
F = np.array([[1, 3],
              [4, 2]])

print('Inverse of F is:\n', np.linalg.inv(F))

# Problem 6
print('\nProblem 6:')
G = np.array([[1, 2, 3, 4, 5, 6],
              [0, 2, 4, 5, 1, 4],
              [0, 1, 4, 9, 7, 2],
              [1, 7, 3, 5, 4, 1],
              [1, 0, 4, 0, 5, 4],
              [1, 8, 7, 3, 1, 0]])

print('Inverse of G is:\n', np.linalg.inv(G))
