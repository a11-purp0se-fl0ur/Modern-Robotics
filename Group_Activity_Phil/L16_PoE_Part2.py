'''
Description: Python code for solving the examples given in ME4140_L16_Product_of_Exponentials_pt2
Author: Phil Deierling
Date: 03/15/2021
Version: 1.02
Log: 
03/15/2021: First submission
03/02/2022: Added another example
02/23/2023: Changed lecture number from 18 to 16
'''

import numpy as np
import sys
import scipy.constants as spc

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
print('Solutions to Example #1')

# Known
L1 = 350
L2 = 410
L3 = L2
L4 = 136
thetaDeg = np.array([0,-90,0,0,90,0,0])

# home position
M = np.eye(4)
M[2,-1] = L1 + L2 + L3 + L4
print('\Home position M: \n', M)


# With screw axes in the space frame
# Distance q from {s} to each joint axis
q1 = np.array([0,0,0])
q2 = np.array([0,0, L1])
q3 = np.array([0,0,0])
q4 = np.array([0, 0, (L1 + L2)])
q5 = np.array([0,0,0])
q6 = np.array([0, 0, (L1 + L2 + L3)])
q7 = np.array([0,0,0])
q = np.concatenate(([q1], [q2], [q3], [q4], [q5], [q6], [q7]), axis=0)

# Screw axis direction for each joint
sHat1 = np.array([0,0,1])
sHat2 = np.array([1,0,0])
sHat3 = sHat1
sHat4 = sHat2
sHat5 = sHat1
sHat6 = sHat2
sHat7 = sHat1
sHat = np.concatenate(([sHat1], [sHat2], [sHat3], [sHat4], [sHat5], [sHat6], [sHat7]), axis=0)

numJoints = len(sHat)
S = np.zeros([6,numJoints])
for i in range(numJoints):
    S[0:3,i] = sHat[i,:]
    S[3:,i] = np.cross(-sHat[i,:], q[i,:])

print('\nScrew axes in the {s} frame: \n', S)


theta = np.deg2rad(thetaDeg)
T = ME4140.FKinSpace(M, S, theta)
print('\nPosition and pose:\n', np.round(T,3))




#-----------------------------------------------------------------------#
pause = True
if not pause:
	print('#-----------------------------------------------------------------------#')
	print('Solutions to Group Activity')
	# Known
	L1 = 550
	L2 = 300
	L3 = 60
	W1 = 45
	thetaDeg = np.array([0,45,0,-45,0,-90,0]) # in degrees

	############### SOLUTION ###############
	# Converting theta to radians
	theta = np.deg2rad(thetaDeg)
	print('theta = \n', np.shape(theta))
	# Defining the home positon
	M = np.eye(4)
	M[2,-1] = L1 + L2 + L3
	print('\nHome position M: \n', M)


	# Distance q from {b} to each joint axis
	q1 = np.array( [ [0, 0, -(L1 + L2 + L3)] ] )
	q2 = q1
	q3 = q1
	q4 = np.array( [ [W1, 0, -(L2 + L3)] ] )
	q5 = np.array( [ [0, 0, -L3] ] )
	q6 = q5
	q7 = q5
	q = np.concatenate((q1, q2, q3, q4, q5, q6, q7), axis=0)

	# Screw axis direction for each joint 
	sHat1 = np.array( [ [0,0,1] ] )
	sHat2 = np.array( [ [0,1,0] ] )
	sHat3 = sHat1
	sHat4 = sHat2
	sHat5 = sHat1
	sHat6 = sHat2
	sHat7 = sHat1
	sHat = np.concatenate((sHat1, sHat2, sHat3, sHat4, sHat5, sHat6, sHat7), axis=0)
	print('\nsHat = \n', sHat)

	# Linear velocity components for each screw axis
	numJoints = len(sHat)
	B = np.zeros([6,numJoints])
	for i in range(numJoints):
		B[0:3,i] = sHat[i,:]                        # angular component
		#B[3:,i] = np.cross( q[i,:], sHat[i,:] )    # linear component (switching cross product order to get rid of the negative sign)
		B[3:,i] = np.cross( -sHat[i,:], q[i,:] )    # linear component

	print('\nScrew axes in the body frame B: \n', B)


	
	TBody = ME4140.FKinBody(M, B, theta)
	print('\nT with body screws:\n', TBody)


