from matplotlib import pyplot as plt
from Functions.Mia_Functions import *
dec = 3
np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------------------------------------------------------------------------------
# Example 1

L1 = L2 = L3 = L4 = 1.5

# Find the Jacobian

# thetas
theta = np.array([np.pi/4, np.pi/-4, np.pi/-2, 0])

# home position
R = np.eye(3)
p = np.array([L1+L2+L3+L4, 0, 0])
M = constructT(R, p)

# Find screws
h = 0
sHat = np.array([0, 0, 1])
q1 = np.array([0, 0, 0])
q2 = np.array([L1, 0, 0])
q3 = np.array([L1+L2, 0, 0])
q4 = np.array([L1+L2+L3, 0, 0])
s1 = parametersToScrew(sHat, q1, h)
s2 = parametersToScrew(sHat, q2, h)
s3 = parametersToScrew(sHat, q3, h)
s4 = parametersToScrew(sHat, q4, h)
S = np.column_stack((s1, s2, s3, s4))

# Jacobian
J = SpaceJacobian(S, theta)
print('\nJ:\n', J)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Example 2

J = np.array([[1, 3], [2, 4], [3, 3]])
pseudoJ = pseudoInv(J)
print('\nInverseJ\n', pseudoJ)

# ----------------------------------------------------------------------------------------------------------------------
# Example 3
J = np.array([ [0, 0, 0, 0.966, 0, 0.256],
               [0, 1, 1, 0, 1, 0],
               [1, 0, 0, -0.259, 0, -0.966],
               [0, -0.675, -1.488, 0, -1.138, 0],
               [0, 0, 0, 1.277, 0, 0.957],
               [0, 0.35, -0.463, 0, 0.685, 0]
               ])
pseudoJ = pseudoInv(J)
print('\nInverseJ\n', pseudoJ)

