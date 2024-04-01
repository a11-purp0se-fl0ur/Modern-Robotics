"""
Description: Various Functions created for ME 4140
Author: Mia Scoblic
Date: 02/03/2024
"""

import numpy as np
from scipy.linalg import expm

# ----------------------------------------------------------------------------------------------------------------------
# Description: Family of functions to calculate rotation matrices
# ----------------------------------------------------------------------------------------------------------------------

# Description: Combining three rotation vectors into a rotation matrix
def rotCombine(x, y, z):
    R = np.column_stack((x, y, z))
    detR = np.linalg.det(R)
    if detR == 1:
        return R
    else:
        raise ValueError("Incorrect Rotation Matrix. The determinant must be equal to 1. Check input.")


# ----------------------------------------------------------------------------------------------------------------------
# Description: Angular Velocity
# ----------------------------------------------------------------------------------------------------------------------

# Description: Convert a vector into a skew-symmetric matrix
def skew(x):
    if len(x) == 3:
        x1, x2, x3 = x
        return np.array([[0, -x3, x2],
                         [x3, 0, -x1],
                         [-x2, x1, 0]])
    elif len(x) == 6:
        x1, x2, x3, x4, x5, x6 = x
        return np.array([
            [0, -x3, x2, x4],
            [x3, 0, -x1, x5],
            [-x2, x1, 0, x6],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Input vector must be either a 3-vector or a 6-vector.")


# Description: Convert the matrix back into a vector
def unSkew(R):
    w1 = R[2, 1]
    w2 = R[0, 2]
    w3 = R[1, 0]
    V = np.row_stack((w1, w2, w3))
    return V


# ----------------------------------------------------------------------------------------------------------------------
# Description: Exponential Coordinates
# ----------------------------------------------------------------------------------------------------------------------

# Description: Calculate R given omega (matrix) and theta, look up pre and post multiplication (need to adjust function if its not in space frame)
def Rod(theta, skewOmega):
    R = np.eye(3) + (np.sin(theta) * skewOmega) + ((1 - np.cos(theta)) * (skewOmega @ skewOmega))
    return R


# Description: Calculate R given omega and theta in vector form
def expCoord_to_R(expCoord):
    theta = np.linalg.norm(expCoord)
    omega = expCoord / theta
    omegaskew = skew(omega)
    R = Rod(theta, omegaskew)
    return R


def matrix_exponential(A, theta, n=50):
    """
    Compute the matrix exponential e^(A*theta) using Taylor series expansion.

    :param A: The matrix
    :param theta: The scalar multiplier
    :param n: Number of terms in the Taylor series expansion
    :return: The matrix exponential of A*theta
    """
    result = np.eye(A.shape[0])  # Start with the identity matrix
    A_scaled = A * theta
    factorial = 1  # Factorial starts at 1! for the first term

    for i in range(1, n):
        factorial *= i
        result += np.linalg.matrix_power(A_scaled, i) / factorial

    return result


# ----------------------------------------------------------------------------------------------------------------------
# Description: Matrix Logarithms
# ----------------------------------------------------------------------------------------------------------------------

# Description: Find axis of rotation and angle of rotation to get to a known R ending point
def Matrix_Logarithm_Rotations(R):
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
        return theta, omega

    # Otherwise, use the general formula
    theta = np.arccos(0.5 * (trR - 1))
    omega = 1 / (2 * np.sin(theta)) * (R - R.T)
    return theta, omega


# ----------------------------------------------------------------------------------------------------------------------
# Description: Transformation Matrices
# ----------------------------------------------------------------------------------------------------------------------

# Description: Construct the T matrix from the R matrix and translation vector
def constructT(R, p):
    # Initialize transformation matrix
    T = np.zeros([4, 4])
    T[0:3, 0:3] = R
    T[:3, 3] = p
    T[-1, -1] = 1
    return T

# Description: Swap subscripts on a transformation matrix.
def invertT(T):
    R = T[0:3, 0:3]
    p = T[:3, 3]

    R_transpose = np.transpose(R)
    neg_R_transpose_p = -R_transpose @ p

    T_inv = np.zeros([4, 4])
    T_inv[0:3, 0:3] = R_transpose
    T_inv[:3, 3] = neg_R_transpose_p
    return T_inv


# ----------------------------------------------------------------------------------------------------------------------
# Description: Twists and Screws
# ----------------------------------------------------------------------------------------------------------------------

# Description: Calculate the adjoint representation of T
def adjoint(T=None, R=None, p=None):
    if T is not None:
        R = T[0:3, 0:3]
        p = T[0:3, 3]
    elif R is None or p is None:
        raise ValueError('Invalid Pass to Function')

    # Convert three-vector to matrix
    pSkew = skew(p)

    # Matrix multiplication
    pR = pSkew @ R

    # Set up Adjoint Matrix
    adjT = np.zeros([6, 6])
    adjT[0:3, 0:3] = R
    adjT[3:6, 0:3] = pR
    adjT[3:6, 3:6] = R

    return adjT


# Description: Calculate screws axis given screw parameters sHat, q, and h
def parametersToScrew(sHat, q, h):
    # Define Sw and Sv
    Sw = np.round(sHat, 3)
    Sv = np.round((np.cross(-1 * sHat, q) + (h * sHat)), 3)

    # Compile S Matrix
    S = np.zeros(6)
    S[:3] = Sw
    S[3:] = Sv

    return S


# Description: Go from Twist to Screw
def twistToScrew(V):
    # Split the vector into linear and angular parts
    Vw = V[:3]  # Keep it as 1D array
    Vv = V[3:]  # Keep it as 1D array

    # Case 1 (Rotation and Translation)
    if np.any(Vw != 0):  # np.any is more appropriate here
        thetaDot1 = np.linalg.norm(Vw)
        S1 = V / thetaDot1
        return S1.flatten()  # Flatten the array to make it 1D
    # Case 2 (Pure Translation)
    else:
        thetaDot2 = np.linalg.norm(Vv)
        S2 = V / thetaDot2
        return S2.flatten()  # Flatten the array to make it 1D



# Description: Go from Screw to Screw Parameters
def screwToParameters(S):
    Sw = S[:3]
    Sv = S[3:]

    # Case 1 (Rotation and Translation)
    if np.all(Sw != 0):
        h = np.round(np.dot(np.transpose(Sw), Sv), 3)
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
# Description: Exponential Coordinates of Rigid Body Motion
# ----------------------------------------------------------------------------------------------------------------------

# Definition: Subfunction to compute G(theta)
def G(theta, Sw_skew):
    I = np.eye(3)
    G_theta = theta * I + (1 - np.cos(theta)) * Sw_skew + (theta - np.sin(theta)) * (Sw_skew @ Sw_skew)
    return G_theta

# Definition: Provide the screw 6-vector in skew-symmetric form, and some theta, to recieve a transformation matrix.
def expCoord_to_T(Screw, Theta, T=None):
    if T is None:
        T = np.eye(4)

    Sw = Screw[:3].flatten()
    Sv = Screw[-3:].flatten()

    Sw_skew = skew(Sw)

    R = Rod(Theta, Sw_skew)

    p = G(Theta, Sw_skew) @ Sv

    #T = np.vstack((np.hstack((R, np.reshape(p, (3, 1)))), [0, 0, 0, 1]))
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    T[3] = [0, 0, 0, 1]  # Ensuring the last row is [0, 0, 0, 1]

    return T

# Definition: Find Screw axis and Theta, given a transformation matrix.

def T_to_Screw(T):
    # Check if T is a 4x4 matrix
    if T.shape != (4, 4):
        raise ValueError("Transformation matrix T must be a 4x4 matrix.")

    R = T[0:3, 0:3]
    p = T[0:3, 3]

    # Check if R is a valid rotation matrix (orthogonal and determinant of 1)
    if not np.allclose(np.dot(R, R.T), np.eye(3)) or not np.isclose(np.linalg.det(R), 1):
        raise ValueError("The upper-left 3x3 part of T must be a valid rotation matrix.")

    trR = np.trace(R)

    # Check if R is the identity matrix (no rotation)
    if np.allclose(R, np.eye(3)):
        theta = 0
        S_w = np.zeros(3)
        if np.linalg.norm(p) != 0:
            S_v = p / np.linalg.norm(p)
        else:
            S_v = np.zeros(3)
    else:
        # Calculate theta
        theta = np.arccos((trR - 1) / 2)
        # Prevent division by a very small number
        if np.isclose(theta, 0):
            # No rotation, or very small rotation
            S_w = np.zeros(3)
            S_v = 0.5 * p  # Approximation for small angles
        else:
            # Compute screw axis
            S_w = 1 / (2 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            G_inv_theta = 1 / theta * np.eye(3) - 0.5 * skew(S_w) + (1 / theta - 0.5 / np.tan(theta / 2)) * np.dot(skew(S_w), skew(S_w))
            S_v = np.dot(G_inv_theta, p)

    S = np.concatenate((S_w, S_v))
    return theta, S

# ----------------------------------------------------------------------------------------------------------------------
# Description: Product of Exponentials
# ----------------------------------------------------------------------------------------------------------------------

# Description: Computes transformation matrix T(theta) in space frame given array of thetas, M, and screws
def PoE_Space(theta, M, screws):
    prod_exp = np.identity(M.shape[0])

    for i in range(theta.shape[1]):
        screw = screws[:, i]
        screw_skew = np.array([[0, -screw[2], screw[1], screw[3]],
                               [screw[2], 0, -screw[0], screw[4]],
                               [-screw[1], screw[0], 0, screw[5]],
                               [0, 0, 0, 0]])
        exp_screw_theta = expm(screw_skew * theta[0, i])
        prod_exp = prod_exp @ exp_screw_theta

    T = prod_exp @ M
    return T

# ----------------------------------------------------------------------------------------------------------------------
# Description: Jacobian
# ----------------------------------------------------------------------------------------------------------------------

# Description: Calculate the Jacobian given array of link lengths and joint angles
def Jacobian(L1, L2, theta1, theta2):
    J = np.array([[-L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2), -L2 * np.sin(theta1 + theta2)], [L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2), L2 * np.cos(theta1 + theta2)]])

    Rank_J = np.linalg.matrix_rank(J)
    print('Rank of the Jacobian: ', Rank_J)
    det_J = np.linalg.det(J)

    if (det_J):
        print('Jacobian is NOT singular')
        print('Determinant is: ', det_J)
    else:
        print('Jacobian is singular')
        print('Determinant is: ', det_J)

    return J

# ----------------------------------------------------------------------------------------------------------------------
# Description: Extra functionality functions
# ----------------------------------------------------------------------------------------------------------------------

# Description: Normalize an input vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# ----------------------------------------------------------------------------------------------------------------------
# Description: Phil Functions
# ----------------------------------------------------------------------------------------------------------------------
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
