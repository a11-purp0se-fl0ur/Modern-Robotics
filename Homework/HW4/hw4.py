"""
Description: ME:4140 Homework 4
Name: Mia Scoblic
Date: 2024-02-15
"""

# Inserting My Functions (scroll down to see problems)------------------------------------------------------------------
def skew(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    aSkew = np.array([[0, -x3, x2],[x3, 0, -x1],[-x2, x1, 0]])
    return aSkew

def unSkew(R):
    w1 = R[2,1]
    w2 = R[0,2]
    w3 = R[1,0]
    V = np.row_stack((w1, w2, w3))
    return V

def expCoord_to_R(expCoord):
    theta = np.linalg.norm(expCoord)
    omega = expCoord / theta
    omegaskew = skew(omega)
    R = Rod(theta, omegaskew)
    return R-


def Rod(theta, skewOmega):
    R = np.eye(3) + (np.sin(theta)*skewOmega) + ((1-np.cos(theta)) * (skewOmega @ skewOmega))
    return R

def Matrix_Logarithm(R):
    # Check for identity matrix
    if np.allclose(R, np.eye(3)):
        theta = 0
        print("The provided matrix was the Identity Matrix. Omega is undefinted.")
        return theta

    # Check for trace -1
    trR = np.trace(R)
    if np.isclose(trR, -1):
        theta = np.pi
        # Select the appropriate omega calculation based on the matrix entries
        if (1 + R[2, 2]) > np.finfo(float).eps:
            omega = 1 / np.sqrt(2 * (1 + R[2, 2])) * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
        elif (1 + R[1, 1]) > np.finfo(float).eps:
            omega = 1 / np.sqrt(2 * (1 + R[1, 1])) * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
        else:
            omega = 1 / np.sqrt(2 * (1 + R[0, 0])) * np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
        thetaRound = np.round(theta, 3)
        omegaRound = np.round(omega, 3)
        return thetaRound, omegaRound

    # Otherwise, use the general formula
    theta = np.arccos(0.5 * (trR - 1))
    omega = 1 / (2 * np.sin(theta)) * (R - R.T)
    thetaRound = np.round(theta, 3)
    omegaRound = np.round(omega, 3)
    return thetaRound, omegaRound

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

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

import numpy as np

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')

Rsb = Rot('z', 45, 'deg') @ Rot('x', 60, 'deg') @ Rot('y', 30, 'deg')
RsbRound = np.round(Rsb,3)
print('Rsb:\n', RsbRound)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:')

theta1, omega1 = Matrix_Logarithm(Rsb)
omega1unskew = unSkew(omega1)
exponentialCoord1 = np.round(omega1unskew*theta1,3)
print('Eponential Coordinates:\n', exponentialCoord1)

# Problem 3 ------------------------------------------------------------------------------------------------------------
print('\nProblem 3:')

Ws = np.array([1, 2, 3])
Rbs = np.transpose(Rsb)
Wb = np.round(Rbs @ Ws, 3)
print('Wb:\n',Wb)

# Problem 4 ------------------------------------------------------------------------------------------------------------
print('\nProblem 4:')

omega2 = np.array([0.267, 0.535, 0.802])
theta2 = np.radians(45)
exponentialCoord2 = np.round(omega2*theta2,3)
print('Exponential Coordinates:\n', exponentialCoord2)

# Problem 5 ------------------------------------------------------------------------------------------------------------
print('\nProblem 5:')
omega2skew = skew(omega2)
R1 = np.round(Rod(theta2,omega2skew),3)
print('Resulting Matrix:\n', R1)

# Problem 6 ------------------------------------------------------------------------------------------------------------
print('\nProblem 6:')

R2 = Rot('y', np.pi/2, 'rad') @ Rot('z', np.pi, 'rad') @ Rot('x', np.pi/2, 'rad')
theta3, omega3 = Matrix_Logarithm(R2)
omega3unskew = np.round(unSkew(omega3),3)
print('Axis of Rotation:\n',omega3unskew)

# Problem 7 ------------------------------------------------------------------------------------------------------------
print('\nProblem 7:')

print('Angle:\n', theta3, 'radians')

# Problem 8 ------------------------------------------------------------------------------------------------------------
print('\nProblem 8:')

print('Exponential Coordinates:\n', omega3unskew*theta3)

# Problem 9 ------------------------------------------------------------------------------------------------------------
print('\nProblem 9:')
expCoord = np.array([1, 2, 1])
R = np.round(expCoord_to_R(expCoord),3)
print('Rotation Matrix:\n',R)