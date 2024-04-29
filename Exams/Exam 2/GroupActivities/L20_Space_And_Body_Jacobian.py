from scipy.optimize._nonlin import Jacobian

from Functions.Mia_Functions import *
import modern_robotics as mr
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:\n')

# Given:
b = 0.05
L1 = L2 = L3 = 0.1
theta_rad = np.array([np.pi/3, np.pi/2, np.pi/3])

# Home Position
R = np.eye(3)
p = np.array([0, L2+L3, b+L1])
M = constructT(R, p)

# Screws
h = 0
sHat1 = np.array([0, 0, 1])
sHat2 = sHat3 = np.array([1, 0, 0])
q1 = np.array([0, 0, b])
q2 = np.array([0, 0, b+L1])
q3 = np.array([0, L2, b+L1])
s1 = parametersToScrew(sHat1, q1, h)
s2 = parametersToScrew(sHat2, q2, h)
s3 = parametersToScrew(sHat3, q3, h)
S = np.column_stack((s1, s2, s3))

# Jacobian
Js = SpaceJacobian(S, theta_rad)
print('Js:\n', Js)