# Make sure this works for planar robot
# Might not have inverse kinematics
# Copy paste long answer

from Functions.Mia_Functions import *
import modern_robotics as mr

dec = 5
np.set_printoptions(precision=5, suppress=True)

L1 = 350
L2 = L3 = 410
L4 = 136

# Home Position
R = np.eye(3)
p = np.array([0, 0, L1+L2+L3+L4])
M = constructT(R, p)
print('Home:\n', M)

# Screws
h = 0
sHat246 = np.array([1, 0, 0])
sHat1357 = np.array([0, 0, 1])
q1 = q3 = q5 = q7 = np.array([0, 0, 0])
q2 = np.array([0, 0, L1])
q4 = np.array([0, 0, L1+L2])
q6 = np.array([0, 0, L1+L2+L3])

s1 = parametersToScrew(sHat1357, q1, h)
s2 = parametersToScrew(sHat246, q2, h)
s3 = parametersToScrew(sHat1357, q3, h)
s4 = parametersToScrew(sHat246, q4, h)
s5 = parametersToScrew(sHat1357, q5, h)
s6 = parametersToScrew(sHat246, q6, h)
s7 = parametersToScrew(sHat1357, q7, h)

Ss = np.column_stack((s1, s2, s3, s4, s5, s6, s7))
print('Ss:\n', Ss)

# Move to {ee}
Sb = adjoint(np.linalg.pinv(M)) @ Ss
print('Sb:\n', Sb)

# Desired Config
Tsd = np.array([[0.9699, -0.2397, 0.0437, -28.6887],[0.2098, 0.913, 0.3499, 514.6592],[-0.1238, -0.3302, 0.9358, 1146.5462],[0, 0, 0, 1]])
print('Tsd:\n', Tsd)

# Initial Guess
theta_deg = np.array([0, 0, 0, 0, 0, 0, 0])
theta_rad = np.radians(theta_deg)

# Initialization
eomg = 1e-3
ev = 1e-3

# Newton Raphson
[theta, success] = mr.IKinBody(Sb, M, Tsd, theta_rad, eomg, ev)
print('\nThetas:\n', np.degrees(theta))
print('\nStatus:\n', success)

# Check
test = mr.FKinBody(M, Sb, theta)
print('Actual:\n', test)
print('Desired:\n', Tsd)

# Problem 2 --------------
print('\nProblem 2:\n')

A = 400
B = 600
C = 1200
D = 37
E = 1250
F = 250

theta_deg = np.array([45, -30, 30, 0, 0, 0])
theta_rad = np.deg2rad(theta_deg)

# Home Config
R = np.eye(3)
p = np.array([A+E+F, 0, B+C-D])
M = constructT(R, p)

# Screws
h = 0

sHat46 = np.array([1, 0, 0])
sHat235 = np.array([0, 1, 0])
sHat1 = np.array([0, 0, 1])

p1 = np.array([0, 0, 0])
p2 = np.array([A, 0, B])
p3 = np.array([A, 0, B+C])
p4 = np.array([A+E, 0, B+C-D])
p5 = np.array([A+E, 0, B+C-D])
p6 = np.array([A+E+F, 0, B+C-D])

s1 = parametersToScrew(sHat1, p1, h)
s2 = parametersToScrew(sHat235, p2, h)
s3 = parametersToScrew(sHat235, p3, h)
s4 = parametersToScrew(sHat46, p4, h)
s5 = parametersToScrew(sHat235, p5, h)
s6 = parametersToScrew(sHat46, p6, h)

Ss = np.column_stack([s1, s2, s3, s4, s5, s6])

Sb = adjoint(np.linalg.pinv(M)) @ Ss
print('Sb:\n', Sb)

# Desired Config
Tsd = [[-0.357, 0.934, -0.015, 1359.626],
        [0.504, 0.179, -0.845, 514.226],
        [-0.786, -0.31, -0.535, 389.951],
        [0, 0, 0, 1]]
print('Tsd:\n', Tsd)

# Initial Guess
theta_deg = np.array([0, 0, 0, 0, 0, 0])
theta_rad = np.radians(theta_deg)

# Initialization
eomg = 1e-3
ev = 1e-3

# Newton Raphson
[theta, success] = mr.IKinBody(Sb, M, Tsd, theta_rad, eomg, ev)
print('\nThetas:\n', np.degrees(theta))
print('\nStatus:\n', success)