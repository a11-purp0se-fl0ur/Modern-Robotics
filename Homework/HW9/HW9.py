from Functions.Mia_Functions import *
import modern_robotics as mr

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')

# Given
L1 = L2 = L3 = 100
b = 50

Tsd = np.array([[0.707, -0.696, -0.123, -127.5],
                [0.707, 0.696, 0.123, 127.5],
                [0, -0.174, 0.985, 190],
                [0, 0, 0, 1]])

# Home Position
Rsb = np.eye(3)
p = np.array([0, L2+L3, b+L1])
M = constructT(Rsb, p)
print('Home Test:\n', M)

# Screws in {s} frame
h = 0
sHat1 = np.array([0, 0, 1])
sHat23 = np.array([1, 0, 0])
q1 = np.array([0, 0, b])
q2 = np.array([0, 0, b+L1])
q3 = np.array([0, L2, b+L1])
s1 = parametersToScrew(sHat1, q1, h)
s2 = parametersToScrew(sHat23, q2, h)
s3 = parametersToScrew(sHat23, q3, h)
Ss = np.column_stack((s1, s2, s3))
print('Space Screw Test:\n', Ss)

# Move to {ee}
Sb = adjoint(np.linalg.pinv(M)) @ Ss
print('Body Screw Test:\n', Sb)

# Initial Guess
theta_deg = np.array([40, 30, -40])
theta_rad = np.radians(theta_deg)

# Initialization
eomg = 1e-3
ev = 1e-3

# Newton Raphson
[theta, success] = mr.IKinBody(Sb, M, Tsd, theta_rad, eomg, ev)
print('\nThetas:\n', np.degrees(theta))
print('\nStatus:\n', success)

test = mr.FKinBody(M, Sb, theta)
print('Actual:\n', test)
print('Desired:\n', Tsd)
# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:')

# Given
Tsd = np.array([[-0.5, -0.707, 0.5, -0.12],
                [0.707, 0, 0.707, 2.295],
                [-0.5, 0.707, 0.5, 1.51],
                [0, 0, 0, 1]])


A = 350
B = 675
C = 1150
D = 41
E = 1200
F = 240

# Find the joint angles to achieve Tsd

# Home Configuration
Rsb = np.eye(3)
p = np.array([A+E+F, 0, B+C-D])
M = constructT(Rsb, p)

# Screws in the {s} frame
h = 0
sHat1 = np.array([0, 0, 1])
sHat235 = np.array([0, 1, 0])
sHat46 = np.array([1, 0, 0])
q1 = np.array([0, 0, 0])
q2 = np.array([A, 0, B])
q3 = np.array([A, 0, B+C])
q4 = np.array([A+E, 0, B+C-D])
q5 = np.array([A+E, 0, B+C-D])
q6 = np.array([A+E+F, 0, B+C-D])

S1 = parametersToScrew(sHat1, q1, h)
S2 = parametersToScrew(sHat235, q2, h)
S3 = parametersToScrew(sHat235, q3, h)
S4 = parametersToScrew(sHat46, q4, h)
S5 = parametersToScrew(sHat235, q5, h)
S6 = parametersToScrew(sHat46, q6, h)
Ss = np.column_stack((S1, S2, S3, S4, S5, S6))

# Screws in {ee} frame
Sb = adjoint(np.linalg.pinv(M)) @ Ss

# Initial Guess
theta_deg = np.array([10, 10, 10, 10, 10, 10])
theta_rad = np.radians(theta_deg)

# Initialization
eomg = 1e-3
ev = 1e-3

# Newton Raphson
[theta, success] = mr.IKinBody(Sb, M, Tsd, theta_rad, eomg, ev)
print('\nThetas:\n', np.degrees(theta))
print('\nStatus:\n', success)

test = mr.FKinBody(M, Sb, theta)
print('Actual:\n', test)
print('Desired:\n', Tsd)

# Problem 3 ------------------------------------------------------------------------------------------------------------
print('\nProblem 3:')

# Given
Tsd = np.array([[0.184, 0.387, 0.904, 1.84],
                [0.089, 0.909, -0.407, 0.021],
                [-0.979, 0.155, 0.133, 0.12],
                [0, 0, 0, 1]])

[theta, success] = mr.IKinBody(Sb, M, Tsd, theta_rad, eomg, ev)
print('\nThetas:\n', np.degrees(theta))
print('\nStatus:\n', success)

test = mr.FKinBody(M, Sb, theta)
print('Actual:\n', test)
print('Desired:\n', Tsd)
