"""
Description: ME:4140 Homework 4
Name: Mia Scoblic
Date: 2024-02-15
"""

import numpy as np
from Functions.Mia_Functions import *
from Functions.Phil_Functions import *

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')

Rsb = Rot('z', 45, 'deg') @ Rot('x', 60, 'deg') @ Rot('y', 30, 'deg')
RsbRound = np.round(Rsb,3)
print('Rsb:\n', RsbRound)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:')

theta1, omega1 = Matrix_Logarithm(Rsb)
omega1unskew = unSkew(omega1)
exponentialCoord1 = np.round(omega1unskew*theta1,3)
print('Eponential Coordinates:\n', exponentialCoord1)

# Problem 3 ------------------------------------------------------------------------------------------------------------
print('\nProblem 3:')

Ws = np.array([1, 2, 3])
Rbs = np.transpose(Rsb)
Wb = np.round(Rbs @ Ws, 3)
print('Wb:\n',Wb)

# Problem 4 ------------------------------------------------------------------------------------------------------------
print('\nProblem 4:')

omega2 = np.array([0.267, 0.535, 0.802])
theta2 = np.radians(45)
exponentialCoord2 = np.round(omega2*theta2,3)
print('Exponential Coordinates:\n', exponentialCoord2)

# Problem 5 ------------------------------------------------------------------------------------------------------------
print('\nProblem 5:')
omega2skew = skew(omega2)
R1 = np.round(Rod(theta2,omega2skew),3)
print('Resulting Matrix:\n', R1)

# Problem 6 ------------------------------------------------------------------------------------------------------------
print('\nProblem 6:')

R2 = Rot('y', np.pi/2, 'rad') @ Rot('z', np.pi, 'rad') @ Rot('x', np.pi/2, 'rad')
theta3, omega3 = Matrix_Logarithm(R2)
omega3unskew = np.round(unSkew(omega3),3)
print('Axis of Rotation:\n',omega3unskew)

# Problem 7 ------------------------------------------------------------------------------------------------------------
print('\nProblem 7:')

print('Angle:\n', theta3, 'radians')

# Problem 8 ------------------------------------------------------------------------------------------------------------
print('\nProblem 8:')

print('Exponential Coordinates:\n', omega3unskew*theta3)

# Problem 9 ------------------------------------------------------------------------------------------------------------
print('\nProblem 9:')
expCoord = np.array([1, 2, 1])
R = np.round(expCoord_to_R(expCoord), 3)
print('Rotation Matrix:\n',R)



