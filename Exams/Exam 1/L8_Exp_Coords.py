from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Group Activity
# Given:
Ws_hat = np.array([0, 0.866, 0.5])
theta_s = np.radians(30)

# Find:
# (1) Exponential Coordinates
expCoord = Ws_hat * theta_s
print('Exponential Coordinates:\n',expCoord)

# (2) Rsb
Rsb = expCoord_to_R(expCoord)
print('Rsb:\n', Rsb)