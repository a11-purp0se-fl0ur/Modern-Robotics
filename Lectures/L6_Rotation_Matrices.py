'''
Description: Python code examples for rotation matricies.
Author: Phil Deierling
Date: 01/29/2023
Version: 1.0
Log: 
01/29/2023: First submission
'''

import numpy as np


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
        R = np.array([ [1, 0, 0],
                       [0, np.cos(angle), -np.sin(angle)],
                       [0, np.sin(angle), np.cos(angle)]
                       ])
    elif axis == 'y':
        R = np.array([ [np.cos(angle), 0, np.sin(angle)],
                       [0, 1, 0],
                       [-np.sin(angle), 0, np.cos(angle)]
                       ])
    elif axis == 'z':
        R = np.array([ [np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]
                       ])
    else:
        raise NameError('Valid axis not provided. Options are: x, y or z.')

    return R





R = Rot('x', 90, 'deg')
print('The rotation matrix is:\n', R, '\n')


# Group activity solutions
# Rsa
Rsa = Rot('x', 180, 'deg') @ Rot('y', 90, 'deg') @ np.eye(3)
print('Rsa is:\n', Rsa, '\n')

# Rsb
Rsb = Rot('x', -90, 'deg')
print('Rsb is:\n', Rsb, '\n')

# Rbs
Rbs = Rsb.T
print('Rbs is:\n', Rbs, '\n')

# Rad
Rab = Rsa.T @ Rsb
print('Rab is:\n', np.round(Rab,0), '\n')

# pb
pb = np.array([2,3,5])
ps = Rsb @ pb
print('ps (row vector) is:\n', ps)
print('Shape of ps: ', np.shape(ps), '\n')

ps = ps.reshape(-1,1)
print('ps (column vector) is:\n', ps)
print('Shape of ps: ', np.shape(ps), '\n')



