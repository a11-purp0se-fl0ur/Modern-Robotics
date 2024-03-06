from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Group Activity

# Given:
Rs = np.eye(3)
Rsb = rotCombine(np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, 0, 1]))
Rsc = rotCombine(np.array([0, 0, 1]), np.array([0, -1, 0]), np.array([1, 0, 0]))

pbc = np.array([0, -1, 4])
psb = np.array([1, 3, -2])

# Find:
# (1) Tsc
Tsb = constructT(Rsb, psb)

Rbs = np.transpose(Rsb)
Rbc = Rbs @ Rsc
Tbc = constructT(Rbc, pbc)

Tsc = Tsb @ Tbc
print('Tsc:\n', Tsc)

# (2) Tcs
Tcs = invertT(Tsc)
print('Tcs:\n', Tcs)