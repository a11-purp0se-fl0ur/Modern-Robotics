from Functions.Mia_Functions import *

# Group Activity

# Given:
Vb = np.array([0, 0, 0, 2, -2, 2])
theta = np.pi/4

# If the frame is initially aligned with {s} at I, has a twist and theta given... What is the new configuration of T?

# Step 1: Go from Twist to Screw
Sb = twistToScrew(Vb)

# Step 2: Find T from Exp_Coord Function
T = expCoord_to_T(Sb, theta)
print('\nT: ', T)

# NUMBERS CHECK --------------------------------------------------------------------------------------------------------
Vb = np.array([1, 0, 0, 2, -2, 2])
theta = np.pi/6

Sb = twistToScrew(Vb)

T = expCoord_to_T(Sb, theta)
print('\nT: ', T)