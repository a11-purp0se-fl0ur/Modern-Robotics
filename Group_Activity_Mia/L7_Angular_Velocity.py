# Angular Velocity Group Activity

import numpy as np
from Functions.Phil_Functions import *
from Functions.Mia_Functions import *

Rsb = Rot('z', 30, 'deg') @ Rot('x', 40, 'deg')

# Find Rsb' (Orientation of b frame) -----------------------------------------------------------------------------------

# Original orientation
R = np.eye(3)

# Orientation after rotation
RsbPrime = R @ Rsb

Round = np.round(RsbPrime, 3)
print("Rsb':\n", Round)

# Find Wb' -------------------------------------------------------------------------------------------------------------

# Given angular velocity w.r.t s-frame
Ws = np.array([3, 3, 2])

# Tranposing to cancel through multiplication
Rbs = np.transpose(Rsb)

# Calculating Wb
Wb = Rbs @ Ws

Round2 = np.round(Wb, 3)
print("Wb:\n", Round2)

# If you want to convert from row to column vector
Wb2 = np.reshape(Round2, (-1,1))
print("Wb:\n", Wb2)