'''
Description: Python code for Group Activity #1 in Lecture #21 (ME4140_L22_Statics_of_Open_Chains)
Author: Phil Deierling
Date: 03/18/2022
Version: 1.0
Log: 
03/18/2022: First submission
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
print('\nSolutions to Group Activity #1')
L1 = 1.0
L2 = L1
L3 = L2
L4 = L3

# Home position
ps = np.array([L1+L2+L3+L4, 0, 0])
Rsb = np.eye(3)
M = ME4140.RpToTrans(Rsb,ps)
print('\nHome position M\n', M)


theta = np.zeros(4)


# Define the screw axis in the SPACE frame
# Omega
w1 = np.array([ [0,0,1] ])
w2 = w1
w3 = w2
w4 = w3
sHat = np.concatenate((w1, w2, w3, w4), axis=0).T
print('\nsHat = \n', sHat)


# Distance q from {s} to each joint axis
q1 = np.array([ [0, 0, 0] ])
q2 = np.array([ [L1, 0, 0] ])
q3 = np.array([ [L1+L2, 0, 0] ])
q4 = np.array([ [L1+L2+L3, 0, 0] ])
q = np.concatenate((q1, q2, q3, q4), axis=0).T # column wise 
print('\nq values = \n', q)

# Build each screw axis with the angular and linear velocity components
numJoints = np.shape(sHat)[1]
print('\nNumber of joints = ', numJoints)
S = np.zeros([6,numJoints])
for i in range(numJoints):
    S[0:3,i] = sHat[:,i]                        # angular component
    S[3:,i] = np.cross(-sHat[:,i], q[:,i] )     # linear component
print('\nScrew axes in the SPACE frame S: \n', S)


# Define the SPACE Jacobian 
theta_config = np.array([0, 0, np.pi/2, -np.pi/2])
Js = ME4140.JacobianSpace(S, theta_config)
print('\nSpace Jacobian\n', Js)

# Desired wrench in the SPACE frame
fs = np.array([ [10,10,0] ])
ms = np.array([ [0,0,10] ])
Fs = np.concatenate((ms, fs), axis=1).T
print('\nWrench Fs\n', Fs)

# Determine the joint torques
tau = Js.T @ Fs
print('\nJoint torques Tau\n', tau)

'''
#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Group Activity #1 (numbers check)')

# Desired wrench in the SPACE frame
fs = np.array([ [-10,10,0] ])
ms = np.array([ [0,0,-10] ])
Fs = np.concatenate((ms, fs), axis=1).T
print('\nWrench Fs\n', Fs)

# Determine the joint torques
tau = Js.T @ Fs
print('\nJoint torques Tau\n', tau)
'''