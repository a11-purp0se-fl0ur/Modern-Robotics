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


# Rotation matrices using angle and direction
def Rot(axis, angle, ang_type):
    '''
    Description: Calculates the rotation matrix for a given direction and angle.
    Input: A rotation axis, angle and angle type, axis, angle, ang_type, respectively.
    Return: The rotation matrix describing the current configuration
    Example Input:
        axis = 'z'
        angle = 30
        ang_type = 'deg'
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    '''
    ang_type = ang_type.upper()

    if ang_type == 'DEG':
        angle = np.radians(angle)

    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]
                      ])
    elif axis == 'y':
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]
                      ])
    elif axis == 'z':
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]
                      ])
    else:
        raise NameError('Valid axis not provided. Options are: x, y or z.')

    return R

# --------------------------------------------------------------------
