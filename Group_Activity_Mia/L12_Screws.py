from Functions.Mia_Functions import *
from Functions.Phil_Functions import *

# Group Activity 1 -----------------------------------------------------------------------------------------------------
# Given
sHat = np.array([0.577, 0.577, 0.577])
q = np.array([1, 1, 2])
h = 10

# Find: Screws Axis S
S = parametersToScrew(sHat, q, h)
print("S:\n", S)

# ----------------------------------------------------------------------------------------------------------------------

# Group Activity 2 -----------------------------------------------------------------------------------------------------
# Given
V = np.array([1.091, 2.182, 4.365, 2.183, -3.274, 1.091])
S = twistToScrew(V)
print("Screw:\n",S)
print(S.shape)

h, sHat, q = screwToParameters(S, 'screw')
print(h)
print(sHat)
print(q)