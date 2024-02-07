'''
Description: Support functions for ME4140
Author: Phil Deierling
Date: 01/19/2023
'''

import numpy as np

# Indicate the number of rows and columns for a matrix
def checkShape(matA, matB):
    if np.shape(matA) == np.shape(matB):
        print('Same shape as last time.')
    else:
        print('New shape!')

# Inverting Matrices
def myInv(matrix):
    return np.linalg.inv(matrix)

# Adding Matrices
def addMats(A, B):
    C = A + B
    return C

#--------------------------------------------------------------------