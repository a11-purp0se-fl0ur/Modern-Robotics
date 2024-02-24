import numpy as np
from Functions.Mia_Functions import *

# Define Home Matrix
R = np.eye(3)
p = np.array([200, 0, 0])
M = constructT(R, p)
print("M:\n", M)

# Determine the S screw in space frame
sHat = np.array([0, 0, 1])
h1 = 0
q1 = np.array([100, 0, 0])
S1 = parametersToScrew(sHat, q1, h1)

h2 = 0
q2 = np.array([200, 0, 0])
S2 = parametersToScrew(sHat, q2, h2)

print("S1:\n", S1)
print("S2:\n", S2)
# Compute matrix exponential (pre-multiply home position)
