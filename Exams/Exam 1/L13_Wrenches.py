from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Group Activity
# Given:
Rab = Rot('z', 30, 'deg') @ Rot('x', 30, 'deg')
pab = np.array([-3, 2, 4])
rb = np.array([2, 3, -4])
fb = np.array([-1, 2, 2])

# Find:
# (1) Fb
Fb = Wrench(fb, rb)
print("Fb:\n", Fb)

# (2) Fa
Tab = constructT(Rab, pab)
Tba = invertT(Tab)

Ad_Tba = adjoint(Tba)

Fa = Ad_Tba.T @ Fb
print("Fa:\n", Fa)

