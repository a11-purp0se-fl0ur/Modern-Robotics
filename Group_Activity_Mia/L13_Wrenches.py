from Functions.Mia_Functions import *
from Functions.Phil_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Group Activity--------------------------------------------------------------------------------------------------------
# Given
Rab = Rot('z', 30, 'deg') @ Rot('x',30,'deg')
pa = np.array([-3, 2, 4])
rb = np.array([2, 3, -4])
fb = np.array([-1, 2, 2])

# Find: Fb
Fb = Wrench(fb, rb)
print("Fb:\n", Fb)


# Find: Fa
Tab = constructT(Rab, pa)
Tba = np.linalg.inv(Tab)

Ad_Tba = adjoint(Tba)

Fa = Ad_Tba.T @ Fb
print("Fa:\n",Fa)

# Group Activity--------------------------------------------------------------------------------------------------------
# Given
Rab2 = Rot('z', 60, 'deg') @ Rot('x', 45, 'deg')
pa2 = np.array([3, 1, 5])
rb2 = np.array([1, 7, -1])
fb2= np.array([2, -3, -4])

# Find Fb
Fb2 = Wrench(fb2, rb2)
print("Fb2:\n",Fb2)

# Find Fa
Tab2 = constructT(Rab2, pa2)
Tba2 = np.linalg.inv(Tab2)
Ad_Tba2 = adjoint(Tba2)
Fa2 = Ad_Tba2.T @ Fb2
print("Fa2:\n",Fa2)