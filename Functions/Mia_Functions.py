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

# Description: Convert the matrix back into a vector
def unSkew(R):
    w1 = R[2,1]
    w2 = R[0,2]
    w3 = R[1,0]
    V = np.row_stack((w1, w2, w3))
    return V

# ----------------------------------------------------------------------------------------------------------------------
# Description: Exponential Coordinates
# ----------------------------------------------------------------------------------------------------------------------

# Description: Calculate R given omega (matrix) and theta
def Rod(theta, skewOmega):
    R = np.eye(3) + (np.sin(theta)*skewOmega) + ((1-np.cos(theta)) * (skewOmega @ skewOmega))
    return R

# Description: Calculate R given omega and theta in vector form
def expCoord_to_R(expCoord):
    theta = np.linalg.norm(expCoord)
    omega = expCoord / theta
    omegaskew = skew(omega)
    R = Rod(theta, omegaskew)
    return R

# ----------------------------------------------------------------------------------------------------------------------
# Description: Matrix Logarithms
# ----------------------------------------------------------------------------------------------------------------------

# Description: Find axis of rotation and angle of rotation to get to a known R ending point
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
# Description: Transformation Matrices
# ----------------------------------------------------------------------------------------------------------------------

# Description: Construct the T matrix from the R matrix and translation vector
def constructT(R, p):
    # Initialize transformation matrix
    T = np.zeros([4,4])
    T[0:3, 0:3] = R
    T[:3, 3] = p
    T[-1, -1] = 1
    return T

# ----------------------------------------------------------------------------------------------------------------------
# Description: Twists and Screws
# ----------------------------------------------------------------------------------------------------------------------

# Description: Calculate the adjoint representation of T
def adjoint(R, p):

    # Convert three-vector to matrix
    pSkew = skew(p)

    # Matrix multiplication
    pR = pSkew @ R

    # Set up Adjoint Matrix
    adjT = np.zeros([6,6])
    adjT[0:3, 0:3] = R
    adjT[3:6, 0:3] = pR
    adjT[3:6, 3:6] = R

    return adjT

# Description: Calculate screws axis given screw parameters sHat, q, and h
def parametersToScrew(sHat, q, h):
    # Define Sw and Sv
    Sw = np.round(sHat,3)
    Sv = np.round((np.cross(-1*sHat, q) + (h * sHat)),3)

    # Compile S Matrix
    S = np.zeros(6)
    S[:3] = Sw
    S[3:] = Sv

    return S

# Description: Go from Twist to Screw
def twistToScrew(V):
    # Reshape input into 1 column vector
    colV = V.reshape(6,1)

    # Split the vector into linear and angular parts
    Vw = colV[:3]
    Vv = colV[3:]

    # Case 1 (Rotation and Translation)
    if np.all(Vw[:] != 0):
        thetaDot1 = np.round(np.linalg.norm(Vw),3)
        S1 = colV / thetaDot1
        return S1
    # Case 2 (Pure Translation)
    else:
        thetaDot2 = np.round(np.linalg.norm(Vv),3)
        S2 = colV / thetaDot2
        return S2

# Description: Go from Screw to Screw Parameters
def screwToParameters(S):

    Sw = S[:3]
    Sv = S[3:]

    # Case 1 (Rotation and Translation)
    if np.all(Sw != 0):
        h = np.round(np.dot(np.transpose(Sw), Sv),3)
        sHat = Sw
        SVr = Sv - (h * sHat)
        q = np.cross(np.transpose(sHat), np.transpose(SVr))
        return h, sHat, q
    # Case 2 (Pure Translation)
    else:
        print("\nh is infinite.")
        h = 0
        sHat = Sv
        print("\nq is not applicable.")
        q = 0
        return h, sHat, q

# ----------------------------------------------------------------------------------------------------------------------
# Description: Wrenches and Power
# ----------------------------------------------------------------------------------------------------------------------

# Description: Return the wrench given a force and distance
def Wrench(f, r):
    m = np.cross(r, f)
    F = np.concatenate((m, f), axis=0)
    return F

# ----------------------------------------------------------------------------------------------------------------------
# Description: Extra functionality functions
# ----------------------------------------------------------------------------------------------------------------------
