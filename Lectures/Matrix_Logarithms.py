
import numpy as np
from My_Functions.Rotation_Matrices_Functions import *

# In Class Activity
R = Rot('x', np.pi, 'rad') @ Rot('y', np.pi/3, 'rad') @ Rot('z', np.pi/3, 'rad')
print(R)

#Check 1
check1 = np.linalg.norm(R-np.eye(3))
print(check1)

#Check 2
check2 = np.trace(R)
print(check2)

#Check 3
theta=np.arccos(0.5*(np.trace(R)-1))

finalTrace = np.trace(final)
print(finalTrace)