from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Given
L1 = 100
L2 = 100
theta1 = np.radians(25)
theta2 = np.radians(15)
theta = np.column_stack((theta1, theta2))
print('\nTheta:\n',theta)

# Define Home Matrix
R = np.eye(3)
p = np.array([L1 + L2, 0, 0])
M = constructT(R, p)

print('\nHome Matrix (M):\n', M)

# Define Screws in the Space Frame
h = 0  # All Joints are Revolute
sHat1 = np.array([0, 0, 1])
sHat2 = np.array([0, 0, 1])
q1 = np.array([0, 0, 0])
q2 = np.array([L1, 0, 0])

S1 = parametersToScrew(sHat1, q1, h)
S2 = parametersToScrew(sHat2, q2, h)

S = np.column_stack((S1, S2))
print('\nScrews:\n', S)

# Compute T
T = PoE_Space(theta, M, S)
print('\nT:\n', T)