from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Group Activity

# Given:
R = Rot('x', np.pi, 'rad') @ Rot('y', np.pi/3, 'rad') @ Rot('z', np.pi/3, 'rad')

# Find:
# (1) Exponential Coordinates
theta, omega_skew = Matrix_Logarithm(R)

omega_hat = unSkew(omega_skew)

expCoord = omega_hat * theta

print('Exponential Coordinates:\n', expCoord)