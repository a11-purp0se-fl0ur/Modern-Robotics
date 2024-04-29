from scipy.optimize._nonlin import Jacobian

from Functions.Mia_Functions import *
import modern_robotics as mr
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:\n')
L1 = L2 = 1
theta = np.array([0, 0])

# Screws
h = 0
sHat1 = sHat2 = np.array([0, 0, 1])
q1 = np.array([0, 0, 0])
q2 = np.array([L1, 0, 0])
s1 = parametersToScrew(sHat1, q1, h)
s2 = parametersToScrew(sHat2, q2, h)
Ss = np.column_stack((s1, s2))

J = SpaceJacobian(Ss, theta)
print(J)