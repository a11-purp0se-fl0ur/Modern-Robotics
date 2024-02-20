import numpy as np
from Functions.Phil_Functions import  *
from Functions.Mia_Functions import *

Vb = np.array([0, 0, 0, 2, -2, 2])
theta = np.pi/4

Tsb = np.eye(4)

v = Vb[3:]

theta_dot = np.linalg.norm(v)

Sw = np.zeros(3)

Sv = v/theta_dot

# Combines two vectors into one
Sb = np.concatenate((Sw, Sv), axis=0)

# Or
Sb = np.zeros(6)
Sb[3:] = Sv

# Continue
SbB = skew(Sb)

T = np.zeros([4,4])
T[0:3, 0:3] = np.eye(3)
