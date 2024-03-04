from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

L1 = 350
L2 = 410
L3 = 410
L4 = 136

thetaDeg = np.array([0, -90, 0, 0, 90, 0, 0])

M = np.eye(4)
M[2, -1] = L1 + L2 + L3 + L4
print(M)

q1