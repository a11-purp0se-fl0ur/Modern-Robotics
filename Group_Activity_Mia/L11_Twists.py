# Lecture 11 - Twists Group Activity

from Functions.Mia_Functions import *
from Functions.Phil_Functions import *

# Given Rsb and ps
Rsb = Rot('y',np.pi/4, 'rad')
ps = np.array([-1, -2, 0])

# Calculate Tsb
Tsb = np.round(constructT(Rsb, ps),2)
print("Tsb: \n", Tsb)

# Given Vb
Vb = np.array([1, 2, 1, 0, 0, 0])

# Find: Vs
adjT = adjoint(Rsb, ps)
Vs = np.round(adjT @ Vb, 3)
print('Vs:\n', Vs)
