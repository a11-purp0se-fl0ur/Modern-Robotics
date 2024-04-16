import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm, solve
import scipy.constants as spc
import matplotlib.pyplot as plt
from Functions.Mia_Functions import *

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


L1 = 1
L2 = 1

# Home matrix
Rsb = np.eye(3)
p = np.array([L1 + L2, 0, 0])
M = constructT(Rsb, p)
print('\nHome:\n', M)

# Desired configuration
Tsd = np.array([[-0.5, -0.866, 0, 0.366],
                [0.866, -0.5, 0, 1.366],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
print('\nDesired configuration:\n', Tsd)  # this should corespond to theta=[30,90] degrees

# Robot screws in the space frame
S = np.array([[0, 0],
              [0, 0],
              [1, 1],
              [0, 0],
              [0, -L1],
              [0, 0]])

# Robot screws in the body frame
B = adjoint(np.linalg.inv(M)) @ S

# Initial guess
theta_deg0 = np.array([10, 10])
theta_rad0 = np.deg2rad(theta_deg0)

# Initialization
epsilon_w = 1e-3  # rotational error, rad
epsilon_v = 1e-3  # translational error, m
it = 0
itmax = 100
ew = 1e6
ev = 1e6
frame = 'body'
# frame = 'space'

# Start of algorithm
print('\nSTART OF ALGORITHM')
print('iter\t theta1 (deg)\ttheta2 (deg)\t x\t y\t wz\t vx\t vy\t ew\t\t ev')
while (ew > epsilon_w or ev > epsilon_v) and it <= itmax:

    if frame == 'space':
        # Configuration at current theta
        Tsb = FKinSpace(M, S, theta_rad0)

        Tbs = np.linalg.inv(Tsb)
        Tbd = Tbs @ Tsd

        # Body twist needed to move from {b} to {d}
        Vb = se3ToVec(MatrixLog6(Tbd))

        # Body twist in the space frame (space twist)
        Vs = Adjoint(Tsb) @ Vb

        Js = JacobianSpace(S, theta_rad0)
        Jinv = np.linalg.pinv(Js)

        V = Vs

    elif frame == 'body':

        Tsb = FKinBody(M, B, theta_rad0)
        J = JacobianBody(B, theta_rad0)

        Tbs = np.linalg.inv(Tsb)
        Jinv = np.linalg.pinv(J)

        Tbd = Tbs @ Tsd

        Vb_bracket = MatrixLog6(Tbd)
        Vb = se3ToVec(Vb_bracket)
        V = Vb

    else:
        # compute = False
        print('Please choose an appropriate frame (body or space) for the calculation.')
        break

    # error calculations
    ew = np.linalg.norm([V[0], V[1], V[2]])
    ev = np.linalg.norm([V[3], V[4], V[5]])

    theta_rad1 = theta_rad0 + Jinv @ V

    # End-effector coordinates
    x, y = Tsb[0:2, -1]

    print('{:d}\t {:.5f}\t{:.5f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3e}\t {:.3e}'.format(it, np.rad2deg(
        theta_rad0[0]), np.rad2deg(theta_rad0[1]),
                                                                                                        x, y, V[2],
                                                                                                        V[3], V[4], ew,
                                                                                                        ev))

    it += 1
    theta_rad0 = theta_rad1

print('\n############# Verification #############')
print('Desired end-effector configuration Tsd:\n', Tsd)
if frame == 'space':
    Tsd_verify = FKinSpace(M, S, theta_rad0)
    print('\nComputed end-effector configuration:\n', Tsd_verify)
elif frame == 'body':
    Tsd_verify = FKinBody(M, B, theta_rad0)
    print('\nComputed end-effector configuration:\n', Tsd_verify)

# Plotting the results
x, y = L1 * np.cos(theta_rad0[0]), L1 * np.sin(theta_rad0[0])
link1 = np.array([[0, 0],
                  [x, y]])

x = L1 * np.cos(theta_rad0[0]) + L2 * np.cos(
    theta_rad0[0] + theta_rad0[1])  # using just to establish a point for verification
y = L1 * np.sin(theta_rad0[0]) + L2 * np.sin(theta_rad0[0] + theta_rad0[1])
link2 = np.array([link1[1, :], [x, y]])

fig, axs = plt.subplots(1, 1)
axs.scatter(link1[:, 0], link1[:, 1], color='red')
axs.scatter(link2[:, 0], link2[:, 1], color='red')
axs.plot(link1[:, 0], link1[:, 1], color='black')
axs.plot(link2[:, 0], link2[:, 1], color='black')
axs.set_aspect('equal', adjustable='datalim')
plt.legend()
plt.grid(True)
plt.show()
