from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')

Ss = np.array([0, 0, 0, 0.259, 0.432, 0.864])
theta = np.pi

Tsa_prime = expCoord_to_T(Ss, theta)
print('Tsa\': \n', Tsa_prime)


# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:')

# Given:
theta = np.pi/3                             # Amount of Rotation
q = np.array([2, 4, 1])                     # Point on Screw Axis
s = np.array([np.pi/4, np.pi/6, np.pi/6])   # Screw Axis
h = 4                                       # Pitch

# First we normalize the screw axis
sHat = normalize(s)

# Go from Parameters to Screw
Sb = parametersToScrew(sHat, q, h)

Tsb_prime = expCoord_to_T(Sb, theta)
print('Tsb\': \n', Tsb_prime)

# Problem 3 ------------------------------------------------------------------------------------------------------------
print('\nProblem 3:')

Tsc = np.array([[1, 0, 0, 0], [0, 0, -1, 1], [0, 1, 0, 2], [0, 0, 0, 1]]) # Initial Frame
print()
Vs = np.array([1, 0, 2, -1, -3, 1]) # Twist in space frame

theta = np.pi

# Step 1: Go from twist to screw
Ss = twistToScrew(Vs)

# Step 2: Use screw and theta to find T
Tsc_prime = expCoord_to_T(Ss, theta)
print('Tsc\': \n', Tsc_prime)

# Problem 4 ------------------------------------------------------------------------------------------------------------
print('\nProblem 4:')
