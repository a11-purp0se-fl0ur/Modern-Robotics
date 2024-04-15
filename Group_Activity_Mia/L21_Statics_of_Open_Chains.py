from Functions.Mia_Functions import *
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Given
theta = np.array([0, 0, np.pi/2, np.pi/-2])
force_s = np.array([10, 10, 0])
moment_s = np.array([0, 0, 10])
L1 = L2 = L3 = L4 = 1

# Define Home Position
R = np.eye(3)
p = np.array([L1+L2+L3+L4, 0, 0])
M = constructT(R, p)
print('\nHome Position:\n', M)

# Screws
h = 0
q1 = np.array([0, 0, 0])
q2 = np.array([L1, 0, 0])
q3 = np.array([L1+L2, 0, 0])
q4 = np.array([L1+L2+L3, 0, 0])
sHat = np.array([0, 0, 1])

S1 = parametersToScrew(sHat, q1, h)
S2 = parametersToScrew(sHat, q2, h)
S3 = parametersToScrew(sHat, q3, h)
S4 = parametersToScrew(sHat, q4, h)

# Jacobian
Js1 = S1

# Js2
exp1 = expCoord_to_T(S1, theta[0])
adj1 = adjoint(exp1)
Js2 = adj1 @ S2

# Js3
exp2 = expCoord_to_T(S2, theta[1])
combine2 = exp1 @ exp2
adj2 = adjoint(combine2)
Js3 = adj2 @ S3

# Js4
exp3 = expCoord_to_T(S3, theta[2])
combine3 = exp1 @ exp2 @ exp3
adj3 = adjoint(combine3)
Js4 = adj3 @ S4

J = np.column_stack((Js1, Js2, Js3, Js4))
print('\nJacobian:\n',J)

# Applied Wrench
Fs = np.array([0, 0, 10, 10, 10, 0])

# Joint Torques
torque = J.T @ Fs
print('\nTorques:\n',torque)