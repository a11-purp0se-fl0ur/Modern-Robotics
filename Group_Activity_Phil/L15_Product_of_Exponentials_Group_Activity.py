'''
Description: Python code for solving the group activity in ME4140_L15_Product_of_Exponentials_pt1
Author: Phil Deierling
Date: 02/23/2023
Version: 1.0
Log: 
02/23/2023: First submission
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
print('#-----------------------------------------------------------------------#')
print('Solutions to Group Activity')

# Known
L1 = 100
L2 = 100
thetaDeg = np.array([25,15])

# home position
M = np.eye(4)
M[0,-1] = L1 + L2
print('\nHome position M: \n', M)


# With screw axes in the space frame
# Distance q from {s} to each joint axis
q1 = np.array([0,0,0])
q2 = np.array([L1,0, 0])
q = np.concatenate(([q1], [q2]), axis=0)


# Screw axis direction for each joint
sHat1 = np.array([0,0,1])
sHat2 = np.array([0,0,1])
sHat = np.concatenate(([sHat1], [sHat2]), axis=0)

numJoints = len(sHat)
S = np.zeros([6,numJoints])
for i in range(numJoints):
    S[0:3,i] = sHat[i,:]
    S[3:,i] = np.cross(-sHat[i,:], q[i,:])

print('\nScrew axes in the {s} frame: \n', S)


theta = np.deg2rad(thetaDeg)
T = ME4140.FKinSpace(M, S, theta)
print('\nPosition and pose:\n', np.round(T,3))

'''
# Manual version (still with functions but showing each result for J1 and J2)
se3mat = ME4140.VecTose3(S[:,0]*theta[0])
T1 = ME4140.MatrixExp6(se3mat)
print('T1 = \n', T1)

se3mat = ME4140.VecTose3(S[:,1]*theta[1])
T2 = ME4140.MatrixExp6(se3mat)
print('T2 = \n', T2)

T = T1 @ T2 @ M
print('\nPosition and pose:\n', np.round(T,3))
'''