'''
Description: Functions.py involved with rotation matrices
Author: Mia Scoblic
Date: 02/03/2024
'''

import numpy as np

'''
Input: given x and y vectors
Output: rotation matrix of a in terms of s
'''
def s_a(x_a, y_a):
    z_a = np.cross(x_a, y_a)
    Rsa = np.column_stack((x_a, y_a, z_a))
    return Rsa

'''
Input: given x and y vectors
Output: rotation matrix of s in terms of a
'''
def a_s(x_a, y_a):
    z_a = np.cross(x_a, y_a)
    Ras = np.transpose(np.column_stack((x_a, y_a, z_a)))
    return Ras

'''
Input: Rotation matrices to be multiplied
Output: Rotation matrix 
'''
def a_b(Ras, Rsb):
    Rab = Ras @ Rsb
    return Rab

'''
Input: Given point in space, and the related rotation matrix
Output: Point in other space
'''
def point(p_a, Rba):
    p_b = p_a @ Rba
    return p_b


'''
Author: Phil Deierling
'''
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

