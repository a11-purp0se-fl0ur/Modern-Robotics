# Intro to transformation Matrices (T)
# Describe the rotation of b wrt s (R) - Ignore the fact its been translated
# For the translation, we describe the movement with a vector (3 dimensions) (p)
# Combine p and R into one 4x4 matrix

# Format T = ([[R, p],[0, 1]])

# Uses:
# Represent a rigid body config
# Change the fram of reference of a vector or frame
# Displace a frame or vector

# Most rules apply, but not all: Finding a point ( 3D ) requires different rules to
# multiply 4x4 3x1 (add a 1 to 3x1 to make a 4x1)

#Rsc

import numpy as np
from Functions.Mia_Functions import *

x_c = [0, 0, 1]
y_c = [0, -1, 0]
z_c = [1, 0, 0]

Rsc = s_a(x_c, y_c)

# Code from Phil
# 1: Find Rsb, Rbc
# 2: Construct Tsb and Tbc
# 3: Tsc = Tsb * Tbc
# 4: Tcs = Inverse of Tsc

ps =
pb =

xb =
yb =
zb =
Rsb =

xc =
yc =
zc =
Rbc =

Tsb = np.zeros([4,4])
Tsb[0:3,0:3] = Rsb #Zeroth untill the 3rd (spot 0, 1, and 2)
Tsb[:3,3] = ps #0th, 1st, 2nd row of third column
Tsb[-1,-1] = 1 # Adding a 1 down in the bottom
print()
