'''
Description: Various Functions created for ME 4140
Author: Mia Scoblic
Date: 02/03/2024
'''

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Description: Family of functions to calculate rotation matrices
# ----------------------------------------------------------------------------------------------------------------------

# Description: Calculating the third rotation vector given two others
def thirdVector(x, y):
    z = np.cross(x, y)
    return z


# Description: Combining three rotation vectors into a rotation matrix
def rotCombine(x, y, z):
    R = np.column_stack((x, y, z))
    return R

# ----------------------------------------------------------------------------------------------------------------------
# Description: Angular Velocity
# ----------------------------------------------------------------------------------------------------------------------

# Description: Convert a vector into a skew-symmetric matrix
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
    theta = np.sqrt((expCoord.T * expCoord)[0])
    omega = expCoord / np.sqrt((expCoord.T * expCoord)[0])
    omegaskew = skew(omega)
    R = Rod(theta, omegaskew)
    return R



# ----------------------------------------------------------------------------------------------------------------------
# Description: Exponential Coordinates
# ----------------------------------------------------------------------------------------------------------------------

# Description: Rodrigues' Formula
def Rod(theta, skewOmega):
    R = np.eye(3) + (np.sin(theta)*skewOmega) + ((1-np.cos(theta)) * (skewOmega @ skewOmega))
    return R

# ----------------------------------------------------------------------------------------------------------------------
# Description: Matrix Logarithms
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# Description: Extra functionality functions
# ----------------------------------------------------------------------------------------------------------------------
