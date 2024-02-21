from Functions.Mia_Functions import *
from Functions.Phil_Functions import *

# Group Activity 1 -----------------------------------------------------------------------------------------------------
# Given
print("Problem 1:")
sHat1 = np.array([0.577, 0.577, 0.577])
q1 = np.array([1, 1, 2])
h1 = 10

# Find: Screws Axis S
S1 = parametersToScrew(sHat1, q1, h1)
print("S:\n", S1)

# ----------------------------------------------------------------------------------------------------------------------

# Group Activity 2 -----------------------------------------------------------------------------------------------------
# Given
print("\nProblem 2:")
V2 = np.array([1.091, 2.182, 4.365, 2.183, -3.274, 1.091])
S2 = twistToScrew(V2)
print("S:\n",S2)

h2, sHat2, q2 = screwToParameters(S2)
print("h:\n",h2)
print("sHat:\n",sHat2)
print("q:\n",q2)

# Group Activity 3 -----------------------------------------------------------------------------------------------------
# Given
print("\nProblem 3:")
w3 = np.array([1,2,1])
q3 = np.array([1,1,2])
h3 = 10

# Convert w into sHat
sHat3 = w3/(np.linalg.norm(w3))

# Find: Screw Axis S
S3 = parametersToScrew(sHat3, q3, h3)
print(S3)