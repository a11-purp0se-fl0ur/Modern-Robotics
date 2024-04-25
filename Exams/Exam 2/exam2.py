import numpy as np

from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

A = 400
B = 600
C = 1200
D = 37
E = 1250
F = 250

theta_deg = np.array([45, -30, 30, 0, 0, 0])
theta_rad = np.deg2rad(theta_deg)

# Home Config
R = np.eye(3)
p = np.array([A+E+F, 0, B+C-D])
M = constructT(R, p)

# Screws
h = 0

sHat46 = np.array([1, 0, 0])
sHat235 = np.array([0, 1, 0])
sHat1 = np.array([0, 0, 1])

p1 = np.array([0, 0, 0])
p2 = np.array([A, 0, B])
p3 = np.array([A, 0, B+C])
p4 = np.array([A+E, 0, B+C-D])
p5 = np.array([A+E, 0, B+C-D])
p6 = np.array([A+E+F, 0, B+C-D])

s1 = parametersToScrew(sHat1, p1, h)
s2 = parametersToScrew(sHat235, p2, h)
s3 = parametersToScrew(sHat235, p3, h)
s4 = parametersToScrew(sHat46, p4, h)
s5 = parametersToScrew(sHat235, p5, h)
s6 = parametersToScrew(sHat46, p6, h)

S = np.column_stack([s1, s2, s3, s4, s5, s6])

T = PoE_Space(theta_rad, M, S)
print(T)
