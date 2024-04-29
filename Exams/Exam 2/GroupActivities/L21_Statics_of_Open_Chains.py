from Functions.Mia_Functions import *
import modern_robotics as mr
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:\n')

# Given: Static Equilibrium at

theta_rad = np.array([0, 0, np.pi/2, -np.pi/2])
fs = np.array([10, 10, 0])
ms = np.array([0, 0, 10])
L1 = L2 = L3 = L4 = 1 # meter

# Find: What are the toques experienced by each joint?

# Home Position
R = np.eye(3)
p = np.array([L1+L2+L3+L4, 0, 0])
M = constructT(R, p)

# Screws
h = 0
sHat1 = sHat2 = sHat3 = sHat4 = np.array([0, 0, 1])
q1 = np.array([0, 0, 0])
q2 = np.array([L1, 0, 0])
q3 = np.array([L1+L2, 0, 0])
q4 = np.array([L1+L2+L3, 0, 0])
s1 = parametersToScrew(sHat1, q1, h)
s2 = parametersToScrew(sHat2, q2, h)
s3 = parametersToScrew(sHat3, q3, h)
s4 = parametersToScrew(sHat4, q4, h)
Ss = np.column_stack([s1, s2, s3, s4])

# Wrench
Fs = np.array([ms[0], ms[1], ms[2], fs[0], fs[1], fs[2]])

# Jacobian
Js = SpaceJacobian(Ss, theta_rad)

# Torque
torque = Js.T @ Fs
print('Torque:\n', torque)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:\n')

fs = np.array([-10, 10, 0])
ms = np.array([0, 0, -10])

# Wrench
Fs = np.array([ms[0], ms[1], ms[2], fs[0], fs[1], fs[2]])

# Torque
torque = Js.T @ Fs
print('Torque:\n', torque)