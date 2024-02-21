'''
Description: Python code for solving the group activity in ME4140_L14_Exponential_Coords_Rigid_Body
Author: Phil Deierling
Date: 03/03/2021
Version: 1.02
Log: 
03/03/2021: First submission
02/23/2022: Added a second example with different numbers so students can run a second case. 
02/23/2023: Updated lecture number from 15 to 14
'''

import numpy as np
import sys

# Add the directory where the function for ME4140 are located
try:
    sys.path.append('/home/phil/Storage/Teaching/CourseResources/ME4140/Codes/Python')
except:
    pass

import ME4140_Functions as ME4140


# Number of decimals to round for printing
dec = 3
np.set_printoptions(precision=3, suppress=True)


#-----------------------------------------------------------------------#
print('\n#-----------------------------------------------------------------------#')
print('Solutions to Class Activity')

# Given the twist (in the body frame)
Vb = np.array( [0, 0, 0, 2, -2, 2] )
theta = np.pi/4

# Initial starting positon of {b} (initially aligned with {s})
Tsb = np.eye(4)

# From Vb, we see that w=0, therefore, the screw pitch is infinite (only translation of the frame)
v = Vb[3:] # linear velocity
print('\nLinear velocity, v\n', v)

# Rotational rate
theta_dot = np.linalg.norm(v)
print('\nTheta dot\n', theta_dot)

# Screw axis angular component
Sw = np.zeros(3)

# Screw axis linear component
Sv = v/theta_dot

# Screw axis
Sb = np.concatenate((Sw, Sv), axis=0)
print('\nScrew axis: \n', Sb)

# Bracketed version of the screw axis
SbB = ME4140.VecTose3(Sb)


# Rigid body matrix exponential (without functions)
T = np.zeros([4,4])
T[0:3, 0:3] = np.eye(3)
T[0:3:, 3] = Sv*theta
print('Rigid body matrix exponential\n', T)

Tsb_p = T @ Tsb
print('\nNew configuration of the frame, {b\'} =\n', Tsb_p)


# Rigid body matrix exponential (using functions)
Tsb_p2 = ME4140.MatrixExp6(SbB*theta) @ Tsb
print('\nNew configuration of the frame (using functions), {b\'} =\n', Tsb_p2)



print('\n#-----------------------------------------------------------------------#')
print('Solutions to Class Activity (numbers check)')
Vb = np.array( [1, 0, 0, 2, -2, 2] )
theta = np.pi/6

Tsb = np.eye(4)

w = Vb[:3]
theta_dot = np.linalg.norm(w)
print('\nTheta dot\n', theta_dot)

v = Vb[3:] 
Sw = w/theta_dot
Sv = v/theta_dot

Sb = np.concatenate((Sw, Sv), axis=0)
print('\nScrew axis: \n', Sb)


Sw_bracket = ME4140.VecToso3(Sw)
print('\nSw_bracket:\n', Sw_bracket)
G_theta = np.eye(3)*theta + (1-np.cos(theta))*Sw_bracket + (theta - np.sin(theta))*(Sw_bracket @ Sw_bracket)
print('\nG function:\n', G_theta)


Tsb_p = np.zeros([4,4])
matrix_exp = ME4140.MatrixExp3(Sw_bracket*theta)
print('\nMatrix exp: \n', matrix_exp)

Tsb_p[0:3, 0:3] = matrix_exp
Tsb_p[0:3, 3] = G_theta @ Sv
print('\nNew configuration of the frame, {b\'} =\n', Tsb_p)


# Using functions
SbB = ME4140.VecTose3(Sb)
print('\nScrew axis bracket: \n', SbB)

Tsb_p = np.matmul( ME4140.MatrixExp6(SbB*theta), Tsb)
print('\nNew configuration of the frame (using functions), {b\'} =\n', Tsb_p)
