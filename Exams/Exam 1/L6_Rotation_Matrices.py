import numpy as np

from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Group Activity

# Given:
xa = np.array([0, 0, 1])
ya = np.array([0, -1, 0])

xb = np.array([1, 0, 0])
yb = np.array([0, 0, -1])

pb = np.array([2, 3, 5])

# Find:
# (1) Rsa
za = np.cross(xa, ya)

Rsa = rotCombine(xa, ya, za)
print('Rsa:\n', Rsa)

# (2) Rsb
zb = np.cross(xb, yb)

Rsb = rotCombine(xb, yb, zb)
print('Rsb:\n', Rsb)

# (3) Rbs
Rbs = np.transpose(Rsb)
print('Rbs:\n', Rbs)

# (4) Rab
Ras = np.transpose(Rsa)

Rab = Ras @ Rsb
print('Rab:\n', Rab)

# (5) Change point b to {s} coordinates.
ps = Rsb @ pb
print('Ps:\n', ps)