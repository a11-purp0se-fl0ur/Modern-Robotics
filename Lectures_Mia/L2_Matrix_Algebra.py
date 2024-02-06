'''
Description: Matrix Algebra Review
Author: Mia Scoblic
Date: 02/05/2024
'''

import numpy as np

'''
All Topics:
# Transpose, Symmetric, Identity
# Addition, Subtraction, Multiplication
# Trace, Inverse
'''

# Vectors
print('\nVectors:')
v = np.array([1, 2, 3])
print('Vector v:\n', v)
print('Shape of vector:\n', np.shape(v))

# Matrices
print('\nMatrices:')
M = np.zeros([4, 3])
print('Matrix M:\n', M)

# Transpose -----------------------------------------------------------------------------------------------------------
# Interchanging rows and columns (flipped over diagonal)
# Can be used to switch m x n
print('\nTranspose:')
A = np.array([ [1,2,3], [4,5,6], [7,8,9] ] )
print('Original:\n', A)

AT = np.transpose(A)
print('Transposed:\n',AT)

# Symmetric-------------------------------------------------------------------------------------------------------------
# When the transpose appear the same as the original

# Identity--------------------------------------------------------------------------------------------------------------
# A matrix multiplied by its inverse is equal to the identity matrix
print('\nIdentity:')
B = np.eye(3)
print(B)

# Addition and Subtraction----------------------------------------------------------------------------------------------
# Just + and -

# Multiplication--------------------------------------------------------------------------------------------------------
D = A @ B

# Trace-----------------------------------------------------------------------------------------------------------------
print('\nTrace:')
print('Trace of Identitiy:', np.trace(B))

# Inverse---------------------------------------------------------------------------------------------------------------
print('\nInverse:')
P = np.array([[1, 3, 4], [5, 6, 7], [8, 9, 2]])
print('Matrix P:\n', P)

L = np.linalg.inv(P)
print('The Inverse of P is:\n', L)

# Rounding Answers------------------------------------------------------------------------------------------------------
print('\nRounding Answers:')
J = np.round(L, 1)
print('The rounded result of matrix P is:\n', J)