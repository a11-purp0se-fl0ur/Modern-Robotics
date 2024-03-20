from Functions.Mia_Functions import *

theta1 = np.pi/3
theta2 = np.pi/2
theta3 = np.pi/3

b = 50
L1 = 100
L2 = 100
L3 = 100

# 1. Find Screws
# Joint 1
omega_hat2 = np.array([0, 0, 1])
q1 = np.array([0, 0, 50])
S = parametersToScrew(omega_hat, q1, 0)
print(S)

# Repeat for Joint 2 and 3

# 2. Find Jacobians
# Joint 1
#Js1 = S1

# Joint 2 and 3
# Use adjoint formula from summary slides
