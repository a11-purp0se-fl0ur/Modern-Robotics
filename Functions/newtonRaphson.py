import numpy as np
from scipy.linalg import logm, expm, norm
from Functions.Mia_Functions import *

# Number of decimals to round for printing
dec = 3
np.set_printoptions(precision=3, suppress=True)

L1 = L2 = 1

# Home matrix
Rsb = np.eye(3)
p = np.array([L1+L2, 0, 0])
M = constructT(Rsb, p)

# Robot screws in the space frame
S = np.array([ [0, 0],
               [0, 0],
               [1, 1],
               [0, 0],
               [0, -L1],
               [0, 0] ])

# Robot screws in the body frame
B = adjoint(np.linalg.inv(M)) @ S

# Step 1: Initialization
# Define your desired end-effector configuration T_sd (4x4 matrix)
# and your initial guess θ^0 (n-sized vector for n joints).
T_sd = np.array([[-0.5, -0.866, 0, 0.366],
                 [0.866, -0.5, 0, 1.366],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

theta_deg = np.array([10, 10])
theta_0 = np.radians(theta_deg)

# Define your tolerances and maximum number of iterations
epsilon_w = 1e-3
epsilon_v = 1e-3
i_max = 100

# Start with the initial guess
theta = theta_0.astype(float)
i = 0

# Step 2: Loop
while True:
    # i. Compute T_bs based on the current guess of joint angles (theta)
    # T_bs = ... (use forward kinematics to find T_bs)
    T_bs = PoE_Body(theta, M, B)

    # ii. Compute the body Jacobian J_b
    J_b = BodyJacobian(B, theta)

    # iii. Compute the inverse or pseudo-inverse of the Jacobian
    J_b_inv = pseudoInv(J_b)

    # iv. Compute the difference in transformation T_bd
    T_bs_inv = np.linalg.inv(T_bs)
    T_bd = T_bs_inv @ T_sd

    # v. Compute the body twist V_b
    V_b = logm(T_bd)
    V_b_unskew = unSkew(V_b)

    # vi. Extract the angular and linear components of the body twist
    w_b = V_b_unskew[0:3]
    v_b = V_b_unskew[3:6]

    # vii. Update the guess for the joint angles
    theta += J_b_inv @ V_b_unskew[:, 0]

    # viii. Compute the errors
    w_b_norm = norm(w_b, 'fro')  # Frobenius norm for angular error
    v_b_norm = norm(v_b)  # Euclidean norm for linear error

    print(f"Iteration {i}:")
    print(f"Theta: {theta}")
    print(f"Angular Error: {w_b_norm}, Linear Error: {v_b_norm}")

    # ix. Check if the errors are within the tolerances or if i has reached i_max
    if w_b_norm <= epsilon_w and v_b_norm <= epsilon_v or i >= i_max:
        break

    # x. Increment i
    i += 1

# The final joint angles are stored in theta
print("Final joint angles:", theta)
