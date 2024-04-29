from Functions.Mia_Functions import *
import modern_robotics as mr
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:\n')

# Given:
theta_deg = np.array([25, 15])
theta_rad = np.deg2rad(theta_deg)

L1 = L2 = 100

# Home Position
R = np.eye(3)
p = np.array([L1+L2, 0, 0])
M = constructT(R, p)

# Screws
h = 0
sHat1 = sHat2 = np.array([0, 0, 1])
q1 = np.array([0, 0, 0])
q2 = np.array([L1, 0, 0])

s1 = parametersToScrew(sHat1, q1, h)
s2 = parametersToScrew(sHat2, q2, h)

Ss = np.column_stack((s1, s2))

# Ending Config
T = PoE_Space(theta_rad, M, Ss)
print(T)