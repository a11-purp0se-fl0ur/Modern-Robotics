# Lecture 10 - Transformation Matrices Group Activity

from Functions.Mia_Functions import *

# Origin Translations
ps = np.array([1, 3, -2])
pb = np.array([0, -1, 4])

# B Frame in S
xb = np.array([0, 1, 0])
yb = np.array([-1, 0, 0])
zb = thirdVector(xb, yb)
Rsb = rotCombine(xb, yb, zb)
print("Rsb:\n", Rsb)

# C Frame in B
xc = np.array([0, 0, 1])
yc = np.array([-1, 0, 0])
zc = thirdVector(xc, yc)
Rbc = rotCombine(xc, yc, zc)
print("Rbc:\n", Rbc)

# Constructing Tsb
Tsb = constructT(Rsb, ps)
print("Tsb:\n", Tsb)

# Constructing Tbc
Tbc = constructT(Rbc, pb)
print("Tbc:\n", Tbc)

# Calculating Tsc
Tsc = Tsb @ Tbc
print("Tsc:\n", Tsc)

# Calculating Tcs
Tcs = np.linalg.inv(Tsc)
print("Tcs:\n", Tcs)
