'''
Description: Python code for solving the group activity in ME4140_L10_Transformation_Matrices
Author: Phil Deierling
Date: 02/22/2021
Version: 1.1
Log: 
02/22/2021: First submission
02/15/2023: Updated problem values and lecture number (12 vs 10)
'''

import numpy as np
import ME4140_Functions as ME4140 # You will need to create your own functions file to run this.  


# Given
ps = np.array( [1,3,-2] )
pb = np.array( [0,-1,4] )

# {b} frame in {s}
xb = np.array( [0,1,0] )
yb = np.array( [-1,0,0] )
zb = np.cross(xb,yb)
Rsb = np.array( [xb, yb, zb] ).T
print('\nRsb: \n', Rsb)

# {c} frame in {b}
xc = np.array( [0,0,1] )
yc = np.array( [-1,0,0] )
zc = np.cross(xc,yc)
Rbc = np.array( [xc, yc, zc] ).T
print('\nRbc: \n', Rbc)

# Transformation matrix (Tsb)
Tsb = np.zeros([4,4])
Tsb[0:3,0:3] = Rsb
Tsb[:3,3] = ps
Tsb[-1,-1] = 1
print('\nTsb: \n', Tsb)


Tsb = ME4140.RpToTrans(Rsb,ps)
print('\nTsb (using function): \n', Tsb)

# Transformation matrix (Tbc)
Tbc = ME4140.RpToTrans(Rbc,pb) # Using a function
print('\nTbc: \n', Tbc)

# Transformation matrix (Tsc)
Tsc = Tsb @ Tbc
print('\nPart 1\nTsc: \n', Tsc)

# Transformation matrix (Tcs)
Tcs = ME4140.TransInv(Tsc)
#Tcs = np.linalg.inv(Tsc) # built in numpy function
print('\nPart 2\nTcs: \n', Tcs)
