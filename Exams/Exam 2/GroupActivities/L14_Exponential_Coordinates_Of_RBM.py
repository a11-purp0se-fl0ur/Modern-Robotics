from Functions.Mia_Functions import *
import modern_robotics as mr
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:\n')
# Given:
Vb = np.array([0, 0, 0, 2, -2, 2])
theta = np.pi/4

# Find: Tsb'

# Initial Position
R = np.eye(3)
p = np.array([0, 0, 0])
Tsb = constructT(R, p)

# In Body Frame
Sb = twistToScrew(Vb)

Tsb_prime = Tsb @ expCoord_to_T(Sb, theta)
print('Tsb_prime:\n', Tsb_prime)
# ----------------------------------------------------------------------------------------------------------------------

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:\n')
# Given:
Vb = np.array([1, 0, 0, 2, -2, 2])
theta = np.pi/6

# Find: Tsb'
Sb = twistToScrew(Vb)

Tsb_prime = Tsb @ expCoord_to_T(Sb, theta)
print('Tsb_prime:\n', Tsb_prime)
# ----------------------------------------------------------------------------------------------------------------------
