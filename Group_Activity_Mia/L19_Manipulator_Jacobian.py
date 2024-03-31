from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Lengths
L1 = L2 = 1

# Angles
theta1 = 0
#theta2 = 0
theta2 = np.pi/4
#theta2 = np.pi

# Calculate Jacobian
J = Jacobian(L1, L2, theta1, theta2)

print('\nJacobian:\n', J)