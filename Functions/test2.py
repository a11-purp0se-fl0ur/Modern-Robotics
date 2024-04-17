import numpy as np
from scipy.linalg import logm, expm, norm
from Functions.Mia_Functions import *

# Number of decimals to round for printing
np.set_printoptions(precision=3, suppress=True)

L1 = L2 = 1

# Home matrix
Rsb = np.eye(3)
p = np.array([L1+L2, 0, 0])
M = constructT(Rsb, p)

# Robot screws in the body frame
B = adjoint(np.linalg.inv(M)) @ np.array([[0, 0],
                                          [0, 0],
                                          [1, 1],
                                          [0, 0],
                                          [0, -L1],
                                          [0, 0]])

# Desired end-effector configuration T_sd
T_sd = np.array([[-0.5, -0.866, 0, 0.366],
                 [0.866, -0.5, 0, 1.366],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

# Initial guess in degrees and then converted to radians
theta_deg0 = np.array([10, 10])
theta_rad0 = np.deg2rad(theta_deg0)

# Define tolerances and maximum number of iterations
epsilon_w = 1e-3 # rotational error, rad
epsilon_v = 1e-3 # translational error, m
it = 0
itmax = 100
ew = 1e6
ev = 1e6

# Start the algorithm
#print(f"{'iter':>4s} {'theta1 (deg)':>12s} {'theta2 (deg)':>12s} {'wz':>6s} {'vx':>6s} {'vy':>6s} {'ew':>8s} {'ev':>8s}")
print('iter\t theta1 (deg)\ttheta2 (deg)\t x\t y\t wz\t vx\t vy\t ew\t\t ev')
while (ew > epsilon_w or ev > epsilon_v) and it <= itmax:
    T_sb = PoE_Body(theta_rad0, M, B)
    J_b = BodyJacobian(B, theta_rad0)

    T_bs = np.linalg.inv(T_sb)
    J_b_inv = np.linalg.pinv(J_b)

    T_bd = T_bs @ T_sd

    V_b_bracket = logm(T_bd)
    V_b = unSkew(V_b_bracket)

    ew = np.linalg.norm([V_b[0], V_b[1], V_b[2]])
    ev = np.linalg.norm([V_b[3], V_b[4], V_b[5]])

    theta_rad1 = theta_rad0 + J_b_inv @ V_b[:, 0]

    x, y = T_sb[0:2, -1]

    print('{:d}\t {:.5f}\t{:.5f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3e}\t {:.3e}'.format(
        it,
        np.rad2deg(theta_rad0[0]),  # Already a scalar
        np.rad2deg(theta_rad0[1]),  # Already a scalar
        x.item(),  # Converts numpy array with one element to a scalar
        y.item(),  # Converts numpy array with one element to a scalar
        V_b[2].item(),  # Assuming this might be array-like, convert to scalar
        V_b[3].item(),
        V_b[4].item(),
        ew,
        ev
    ))

    it += 1
    theta_rad0 = theta_rad1

print("Final joint angles:", np.degrees(theta_rad0))
