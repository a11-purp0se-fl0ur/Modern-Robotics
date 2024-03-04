from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('Problem 1:')
omegaHat = np.array([0.577, 0.577, 0.577])
theta = np.radians(45)
Rsb = np.eye(3)

skewOmega = skew(omegaHat)

RsbPrime = Rod(theta, skewOmega)
print('RsbPrime:')
print(RsbPrime)
# ----------------------------------------------------------------------------------------------------------------------

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('Problem 2:')
fs = np.array([50, 0, 0])
rs = np.array([0, 0, 75])

Fs = Wrench(fs, rs)
print(Fs)
# ----------------------------------------------------------------------------------------------------------------------
