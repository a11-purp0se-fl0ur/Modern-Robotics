# Exponential Coordinates Group Activity

import numpy as np
from Functions.Mia_Functions import *

# Given
Ws = np.array([0, 0.866, 0.5])
theta = 0.524

# Determine exponential coordinates
expCoord = Ws * theta
print("Exponential Coordinates:\n", expCoord)

# Determine Rsb

# Creating skew matrix
skewOmega = skew(Ws)

# Construction rotation matrix using Rod Formula
Rsb = Rod(theta, skewOmega)
print("Rsb:\n",Rsb)