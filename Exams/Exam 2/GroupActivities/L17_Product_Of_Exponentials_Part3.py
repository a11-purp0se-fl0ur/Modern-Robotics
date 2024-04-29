from Functions.Mia_Functions import *
import modern_robotics as mr
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:\n')

# Given
W1 = 109
W2 = 82
L1 = 425
L2 = 392
H1 = 89
H2 = 95

theta_deg = np.array([0, -90, 0, 0, 90, 0])
theta_rad = np.deg2rad(theta_deg)

# Home Position
R = np.array([[-1, 0, 0],
              [0, 0, 1],
              [0, 1, 0]])
p = np.array([L1+L2, W1+W2, H1-H2])
M = constructT(R, p)

# Screws
h = 0
sHat1 = np.array([0, 0, 1])
sHat2 = np.array([0, 1, 0])
sHat3 = np.array([0, 1, 0])
sHat4 = np.array([0, 1, 0])
sHat5 = np.array([0, 0, -1])
sHat6 = np.array([0, 1, 0])
q1 = np.array([0, 0, 0])
q2 = np.array([0, 0, H1])
q3 = np.array([L1, 0, H1])
q4 = np.array([L1+L2, 0, H1])
q5 = np.array([L1+L2, W1, 0])
q6 = np.array([L1+L2, 0, H1-H2])

s1 = parametersToScrew(sHat1, q1, h)
s2 = parametersToScrew(sHat2, q2, h)
s3 = parametersToScrew(sHat3, q3, h)
s4 = parametersToScrew(sHat4, q4, h)
s5 = parametersToScrew(sHat5, q5, h)
s6 = parametersToScrew(sHat6, q6, h)

Ss = np.column_stack((s1, s2, s3, s4, s5, s6))

T = PoE_Space(theta_rad, M, Ss)
print('T:\n', T)

# Try Body
Sb = adjoint(np.linalg.pinv(M)) @ Ss
T = PoE_Body(theta_rad, M, Sb)
print(T)