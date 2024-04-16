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

# Description: REDO Matrix Log
def Matrix_Logarithm(matrix):
    """
    Calculate the matrix logarithm for either a 3x3 rotation matrix R or a 4x4 transformation matrix T,
    and return the result as a 4x4 skew-symmetric matrix.

    Parameters:
    - matrix: A numpy array of shape (3,3) or (4,4)

    Returns:
    - log_matrix: A 4x4 skew-symmetric matrix representing the logarithm of the matrix.
    """
    if matrix.shape == (3, 3):  # Handle 3x3 rotation matrix
        R = matrix
        if np.allclose(R, np.eye(3)):
            return np.zeros((4, 4))  # Return zero matrix for identity

        trR = np.trace(R)
        if np.isclose(trR, -1):
            theta = np.pi
            if (1 + R[2, 2]) > np.finfo(float).eps:
                omega = np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
            elif (1 + R[1, 1]) > np.finfo(float).eps:
                omega = np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
            else:
                omega = np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
            omega /= np.sqrt(2 * (1 + omega[-1]))
            omega_skew = skew(omega)
            log_matrix = np.zeros((4, 4))
            log_matrix[:3, :3] = omega_skew
            return log_matrix

        theta = np.arccos(0.5 * (trR - 1))
        omega = 1 / (2 * np.sin(theta)) * (R - R.T)
        omega_skew = skew(omega)
        log_matrix = np.zeros((4, 4))
        log_matrix[:3, :3] = omega_skew
        return log_matrix

    elif matrix.shape == (4, 4):  # Handle 4x4 transformation matrix
        R = matrix[:3, :3]
        p = matrix[:3, 3]
        log_matrix = Matrix_Logarithm(R)  # Recursively use the same function for rotation part
        if np.trace(R) == 3:  # Special case for zero rotation
            log_matrix[0:3, 3] = p
        else:
            theta = np.arccos((np.trace(R) - 1) / 2)
            omega_skew = log_matrix[:3, :3]
            G_inv = np.eye(3) - 0.5 * omega_skew + (1 - 0.5 / np.tan(0.5 * theta)) * omega_skew @ omega_skew
            v = G_inv @ p
            log_matrix[0:3, 3] = v

        return log_matrix

    else:
        raise ValueError("Input must be either a 3x3 rotation matrix or a 4x4 transformation matrix.")


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

# Description: Deconstruct the T matrix to the R matrix and translation vector
def deconstructT(T):
    R = T[0:3, 0:3]
    p = T[:3, 3]
    return R, p

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

# Description: Computes transformation matrix T(theta) in body frame given array of thetas, M, and screws
def PoE_Body(theta, M, screws):
    prod_exp = np.identity(M.shape[0])

    for i in range(theta.size):  # This will iterate over the number of elements in theta
        screw = screws[:, i]
        screw_skew = np.array([[0, -screw[2], screw[1], screw[3]],
                               [screw[2], 0, -screw[0], screw[4]],
                               [-screw[1], screw[0], 0, screw[5]],
                               [0, 0, 0, 0]])
        exp_screw_theta = expm(screw_skew * theta[i])  # Notice we use theta[i] instead of theta[0, i]
        prod_exp = prod_exp @ exp_screw_theta

    T = M @ prod_exp
    return T


# ----------------------------------------------------------------------------------------------------------------------
# Description: Singularities
# ----------------------------------------------------------------------------------------------------------------------

# Description: Code to determine if an input matrix is at a singularity by determining the rank and determinant
def singularity(A):
    if A.shape[0] != A.shape[1]:
        raise ValueError('Matrix is not square.')
    else:
        rank_A = np.linalg.matrix_rank(A)

        det_A = np.linalg.det(A)
        if det_A < 0.0001:
            det_A = 0

        if det_A == 0:
            print('The matrix is less than full rank, at a singularity')
        else:
            print('The matrix is full rank, not singular')

        print('Determinant:', det_A)
        print('Rank:', rank_A)

# ----------------------------------------------------------------------------------------------------------------------
# Description: Jacobian
# ----------------------------------------------------------------------------------------------------------------------

# Description: Calculate the Jacobian in the space frame given screws and thetas
def SpaceJacobian(S, theta):
    """
    Construct the Jacobian matrix for a robot with multiple joints.

    Parameters:
    - S: An (6 x n) matrix of screw axes, where each column corresponds to a screw axis for a joint.
    - theta: A list or array of joint angles, corresponding to each screw axis in S.

    Returns:
    - J: The Jacobian matrix of the robot at the given configuration.
    """
    num_joints = theta.size
    J = np.zeros((6, num_joints))  # Initialize Jacobian matrix with zeros

    # Start with the first column
    J[:, 0] = S[:, 0]

    # Product of exponentials up to the previous joint
    T = np.eye(4)  # Start with the identity transformation

    for i in range(1, num_joints):
        # Compute the transformation matrix from the previous joint's screw axis and angle
        T = T @ expCoord_to_T(S[:, i - 1], theta[i - 1])

        # Calculate the adjoint transformation of T
        Adj_T = adjoint(T)

        # Compute the screw axis for current joint in the space frame
        J[:, i] = Adj_T @ S[:, i]

    return J

# Description: Same thing, but for the body frame
def BodyJacobian(S, theta):
    """
    Construct the Body Jacobian matrix for a robot with multiple joints.

    Parameters:
    - S: An (6 x n) matrix of screw axes in the body frame at the home configuration,
         where each column corresponds to a screw axis for a joint.
    - theta: A list or array of joint angles, corresponding to each screw axis in S.

    Returns:
    - Jb: The body Jacobian matrix of the robot at the given configuration.
    """
    num_joints = theta.size
    Jb = np.zeros((6, num_joints))  # Initialize Jacobian matrix with zeros
    T = np.eye(4)  # Start with the identity transformation

    # Compute the forward transformation from base to end-effector
    for i in range(num_joints):
        T = T @ expCoord_to_T(S[:, i], theta[i])

    # Work backwards from end-effector to base
    for i in reversed(range(num_joints)):
        # Compute the transformation for current joint
        T = T @ expCoord_to_T(S[:, i], -theta[i])

        # Calculate the adjoint transformation of the inverse of T
        Adj_T_inv = adjoint(T)

        # Update the body Jacobian matrix for the current joint
        Jb[:, i] = Adj_T_inv @ S[:, i]

    return Jb

# ----------------------------------------------------------------------------------------------------------------------
# Description: Pseudo Inverse
# ----------------------------------------------------------------------------------------------------------------------

# Decription: Given a jacobian, determine the pseudo inverse depending on specific conditions
def pseudoInv(J):
    if J.shape[0] == J.shape[1]:
        if np.linalg.matrix_rank(J) == J.shape[0]:
            pseudoJ = np.linalg.inv(J)
        else:
            raise ValueError("Jacobian is square but not full rank")
    elif J.shape[0] > J.shape[1]:
        pseudoJ = np.linalg.inv(J.T @ J) @ J.T
    elif J.shape[0] < J.shape[1]:
        pseudoJ = J.T @ np.linalg.inv(J @ J.T)
    else:
        raise ValueError("None of the conditions are met. Check for errors.")
    return pseudoJ

# ----------------------------------------------------------------------------------------------------------------------
# Description: Inverse Kinematics Newton-Raphson
# ----------------------------------------------------------------------------------------------------------------------

# Description: Find array of angles that move robot to desired transformation matrix
def newtonRaphson(M, Tsd, S, theta0, frame):
    # Initialization
    epsilon_w = 1e-3  # rotational error, rad
    epsilon_v = 1e-3  # translational error, m
    it = 0
    itmax = 100
    ew = 1e6
    ev = 1e6

    # Start of algorithm
    print('\nSTART OF ALGORITHM')
    print('iter\t theta1 (deg)\ttheta2 (deg)\t x\t y\t wz\t vx\t vy\t ew\t\t ev')
    while (ew > epsilon_w or ev > epsilon_v) and it <= itmax:
        if frame == 'space':
            Tsb = PoE_Space(theta0, M, S)

            Tbs = np.linalg.inv(Tsb)
            Tbd = Tbs @ Tsd

            Vb = unSkew(Matrix_Logarithm(Tbd))

            Vs = adjoint(Tsb) @ Vb

            Js = SpaceJacobian(S, theta0)
            Jinv = np.linalg.pinv(Js)

            V = Vs

        elif frame == 'body':
            Tsb = PoE_Body(theta0, M, S)
            J = BodyJacobian(S, theta0)

            Tbs = np.linalg.inv(Tsb)
            Jinv = np.linalg.pinv(J)

            Tbd = Tbs @ Tsd

            Vb_bracket = Matrix_Logarithm(Tbd)
            Vb = unSkew(Vb_bracket)
            V = Vb

        else:
            print('Please choose an appropriate frame (body or space) for the calculation.')
            break

        # error calculations
        ew = np.linalg.norm([V[0], V[1], V[2]])
        ev = np.linalg.norm([V[3], V[4], V[5]])

        theta1 = theta0 + Jinv @ V

        # End-effector coordinates
        x, y = Tsb[0:2, -1]

        print('{:d}\t {:.5f}\t{:.5f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3e}\t {:.3e}'.format(it, np.rad2deg(theta0[0]), np.rad2deg(theta0[1]),
                                                                                    x, y, V[2], V[3], V[4], ew, ev))

        it += 1
        theta0 = theta1



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
