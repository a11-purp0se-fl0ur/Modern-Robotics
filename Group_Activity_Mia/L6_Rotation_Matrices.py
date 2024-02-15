# Solution to the Lecture 6 Group Activity

from Functions.Mia_Functions import *
import numpy as np

x_a = matrix("0 0 1")
y_a = matrix("0 -1 0")
z_a = thirdVector(x_a, y_a)

x_b = matrix("1 0 0")
y_b = matrix("0 0 -1")
z_b = thirdVector(x_b, y_b)

# Find Rsa
Rsa = rotCombine(x_a, y_a, z_a)
print("Rsa: \n", Rsa)

# Find Rsb
Rsb = rotCombine(x_b, y_b, z_b)
print("Rsb: \n", Rsb)

# Find Rbs
Rbs = np.transpose(Rsb)
print("Rbs: \n", Rbs)

# Find Rab
Ras = np.transpose(Rsa)
Rab = Ras @ Rsb
print("Rab: \n", Rab)

# Change the point Pb to s coordinates
Pb = matrix("2; 3; 5")
Ps = Rsb @ Pb
print("Ps: \n", Ps)