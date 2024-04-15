'''
Description: Python code for examples of singular matricies and rank and group
activity in Lecture #22 (ME4140_L23_Singularities_and_Manipulability)
Author: Phil Deierling
Date: 03/15/2022
Version: 1.0
Log:
03/015/2022: First submission
'''
import numpy as np
import sys

dec = 3
np.set_printoptions(precision=3, suppress=True)
A = np.array([ [1,-2,3],
[2,-3,5],
[1,1,0]])
#A = A.T
print('\nMatrix A:\n', A)
#print('Matrix A.T:\n', A.T)
rank_A = np.linalg.matrix_rank(A)
print('\nRank of A: ', rank_A, '\n')
det_A = np.linalg.det(A)
print('\nDeterminant of A: ', det_A, '\n')

if (det_A):
    print('Matrix A is NOT singular')
else:
    print('Matrix A is singular')

# Example when the above approach gives a false result
L1 = 1
L2 = 1
theta1 = 0
theta2 = np.pi/4
#theta2 = 3*np.pi/4
#theta2 = 0
#theta2 = np.pi
J = np.array([ [-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2), -L2*np.sin(theta1
+ theta2)],
[L1*np.cos(theta1) + L2*np.cos(theta1 + theta2), L2*np.cos(theta1 +
theta2)] ])
print('Jacobian: \n', J)
Rank_J = np.linalg.matrix_rank(J)
print('Rank of the Jacobian: ', Rank_J)
det_J = np.linalg.det(J)

if (det_J):
    print('Jacobian is NOT singular')
    print('Determinant is: ', det_J)
else:
    print('Jacobian is singular')
    print('Determinant is: ', det_J)
