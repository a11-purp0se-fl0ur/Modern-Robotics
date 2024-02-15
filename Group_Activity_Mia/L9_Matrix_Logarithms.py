# Matrix Logarithms Group Activity

from Functions.Mia_Functions import *
from Functions.Phil_Functions import *
import numpy as np

R = Rot('x', np.pi, 'rad') @ Rot('y', np.pi / 3, 'rad') @ Rot('z', np.pi / 3, 'rad')

# Calculating theda a omega from the given rotation matrix
theta, omega = Matrix_Logarithm(R)

# Unskewing the omega matrix into a vector
omegaUnSkew = unSkew(omega)

# Printing exponential coordinates
print(omegaUnSkew * theta)