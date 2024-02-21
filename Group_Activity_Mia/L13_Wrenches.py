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

Ad_Tba = ad