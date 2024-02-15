'''
Description: Python code for solving the group activity in ME4140_L09_Matrix_Logarithms 
Author: Phil Deierling
Date: 02/05/2024
Version: 1.0
Log: 
02/05/2024: First submission
'''

import numpy as np
import ME4140_Functions as ME4140 # You will need to create your own functions file to run this.                


R = ME4140.Rot('x', np.pi, 'rad') @ ME4140.Rot('y', np.pi/3, 'rad') @ ME4140.Rot('z', np.pi/3, 'rad')
print('The rotation matrix is:\n', np.round(R,3))

# Check 1 (does R = I)
print('\nnorm(R-I) should be close to zero.') # near zero
check1 = np.linalg.norm(R-np.eye(3))
print(check1)

# Check 2 (does trace(R) = -1)
print('\nDoes trace(R) = -1?') 
check2 = np.trace(R)
print(check2)

# Case #3 then
theta = np.arccos(0.5*(np.trace(R)-1))
print('\nTheta:\n', theta)

omega_bracket = (1/(2*np.sin(theta)))*(R-R.T)
print('\nOmega skew-symmetric:\n', omega_bracket)

omega1 = omega_bracket[2,1]
omega2 = omega_bracket[0,2]
omega3 = omega_bracket[1,0]

omega_hat = np.array( [omega1, omega2, omega3] )
print('\nOmega hat (row vec):\n', omega_hat)
print('\nOmega hat (col vec):\n', omega_hat.reshape(-1,1))


exp_coords = np.round(omega_hat*theta,3)
print('\n########## FINAL ANSWER ##########')
print('Exponential coordinates (row vec):\n', exp_coords)
print('\nExponential coordinates (col vec):\n', exp_coords.reshape(-1,1))

# Using a function to do all of this
omega_bracket_x_theta = ME4140.MatrixLog3(R) # I wrote my function to return the exponential coordinates in matrix form.

exp_coords2 = ME4140.so3ToVec(omega_bracket_x_theta) # Function to extract a vector from the skew-symmetric matrix
exp_coords2 = np.round(exp_coords2,3)
print('\nExponential coordinates using functions (col vec):\n', exp_coords2.reshape(-1,1))
    

