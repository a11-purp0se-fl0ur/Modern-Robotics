'''
Description: Python code for solving the group activity in ME4140_L11_Twists
Author: Phil Deierling
Date: 02/26/2021
Version: 1.1
Log: 
02/26/2021: First submission
02/15/2023: Updated problem values and lecture number (13 vs 11)
'''

import numpy as np
import ME4140_Functions as ME4140 # You will need to create your own functions file to run this.  


# Given
Vb = np.array( [1,2,1,0,0,0] )
R = ME4140.Rot('y',np.pi/4, 'Rad')
ps = np.array( [-1,-2,0] )

# Solution
Rsb = R 
print('\nRsb: \n', Rsb)

# Transformation matrix (Tsb)
Tsb = ME4140.RpToTrans(Rsb,ps)
print('\nTsb: \n', Tsb)

# Twist in the space frame
Ad_Tsb = np.zeros([6,6])
Ad_Tsb[0:3, 0:3] = R # the first three rows and columns are the rotation matrix
p_bracket = ME4140.VecToso3(ps)
Ad_Tsb[3:, 0:3] = p_bracket @ R # the last three rows and first three columns are [p]*R
Ad_Tsb[3:, 3:] =  Rsb # the last three rows and columns are Rsb
print('\nAd_Tsb: \n', Ad_Tsb)

Ad_Tsb = ME4140.Adjoint(Tsb) # using a function
print('\nAd_Tsb (from function): \n', Ad_Tsb)

Vs = np.matmul(Ad_Tsb, Vb).reshape(-1,1) # turn the row vector into a column vector
print('\nTwist in space frame: \n', Vs)

