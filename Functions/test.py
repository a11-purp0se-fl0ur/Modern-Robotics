import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm, solve

def MatrixExp6(se3_mat):
    """Compute the matrix exponential of an se(3) matrix."""
    return expm(se3_mat)

def VecTose3(V):
    """Convert a 6-element vector into an se(3) matrix."""
    omega = V[0:3]
    v = V[3:6]
    omega_skew = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    se3_mat = np.zeros((4, 4))
    se3_mat[0:3, 0:3] = omega_skew
    se3_mat[0:3, 3] = v
    return se3_mat

def se3ToVec(se3_mat):
    """
    Convert an se(3) matrix to a 6-dimensional vector (twist vector).
    """
    omega = [se3_mat[2, 1], se3_mat[0, 2], se3_mat[1, 0]]  # Extracts the skew-symmetric part's off-diagonal elements
    v = se3_mat[0:3, 3]                                    # Extracts the translation part
    return np.array(omega + list(v))

def RpToTrans(R, p):
    """Convert rotation matrix and position vector into homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def Adjoint(T):
    """Compute the adjoint representation of a transformation matrix."""
    R = T[:3, :3]
    p = T[:3, 3]
    p_skew = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    Ad_T = np.zeros((6, 6))
    Ad_T[:3, :3] = R
    Ad_T[3:, 3:] = R
    Ad_T[3:, :3] = p_skew @ R
    return Ad_T

def FKinSpace(M, S, theta):
    """Calculate forward kinematics for the space frame."""
    T = M.copy()
    for i in range(len(theta)):
        T = T @ MatrixExp6(VecTose3(S[:, i] * theta[i]))
    return T

def FKinBody(M, B, theta):
    """Calculate forward kinematics for the body frame."""
    T = M.copy()
    for i in range(len(theta)-1, -1, -1):
        T = MatrixExp6(VecTose3(B[:, i] * theta[i])) @ T
    return T

def JacobianSpace(S, theta):
    """Calculate space Jacobian."""
    J = np.zeros((6, len(theta)))
    T = np.eye(4)
    for i in range(len(theta)):
        J[:, i] = Adjoint(np.linalg.inv(T)) @ S[:, i]
        T = T @ MatrixExp6(VecTose3(S[:, i] * theta[i]))
    return J

def JacobianBody(B, theta):
    """Calculate body Jacobian."""
    J = np.zeros((6, len(theta)))
    T = np.eye(4)
    for i in range(len(theta)):
        T = T @ MatrixExp6(VecTose3(B[:, i] * theta[i]))
    for i in range(len(theta)):
        J[:, i] = Adjoint(T) @ B[:, i]
        T = T @ MatrixExp6(VecTose3(B[:, i] * -theta[i]))
    return J

def MatrixLog6(T):
    """Calculate the matrix logarithm of a homogeneous transformation matrix."""
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    if np.allclose(R, np.eye(3)):  # Special case for zero rotation where acos can be unstable
        return np.block([[np.zeros((3, 3)), np.reshape(p, (3, 1))], [0, 0, 0, 0]])

    theta = np.arccos(max(min((np.trace(R) - 1) / 2, 1), -1))  # Clamping the acos input within valid range
    if np.isclose(theta, 0):  # Infinitesimally small rotation
        w_skew = (R - np.eye(3)) / 2
    else:
        w_skew = theta / (2 * np.sin(theta)) * (R - R.T)

    w = np.array([w_skew[2, 1], w_skew[0, 2], w_skew[1, 0]])
    G_inv = np.eye(3) - 0.5 * w_skew + (1 - theta / (2 * np.tan(theta / 2))) / theta * (np.outer(w, w))
    v = solve(G_inv, p)

    se3_mat = np.zeros((4, 4))
    se3_mat[0:3, 0:3] = w_skew
    se3_mat[0:3, 3] = v
    se3_mat[3, 3] = 0  # The last row remains unchanged

    return se3_mat
