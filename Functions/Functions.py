'''
Description: Support functions for ME4140_L02_Matrix_Algebra_Review
Author: Phil Deierling
Date: 01/19/2023
Version: 1.0
Log:
01/19/2023: First submission
'''

import numpy as np


def checkShape(matA, matB):
    if np.shape(matA) == np.shape(matB):
        print('Same shape as last time.')
    else:
        print('New shape!')


def myInv(matrix):
    return np.linalg.inv(matrix)

def addMats(A, B):
    C = A + B
    return C

#--------------------------------------------------------------------