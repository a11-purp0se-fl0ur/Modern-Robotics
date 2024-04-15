'''
Description: Python code for solving examples in Lecture #26 (ME4140_L26_Numerical_Inverse_Kinematics_pt2)
Author: Phil Deierling
Date: 04/20/2022
Version: 1.0
Log: 
04/20/2022: First submission
04/21/2023: Added 
'''

import numpy as np
import sys
import scipy.constants as spc
import matplotlib.pyplot as plt


# Add the directory where the function for ME4140 are located
try:
    sys.path.append('/home/phil/Documents/Teaching/CourseResources/ME4140/Codes/Python')
except:
    pass
import ME4140_Functions as ME4140


# Number of decimals to round for printing
dec = 3
np.set_printoptions(precision=3, suppress=True)

#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Group Activity #1 (Coding example for Space Jacobian)')
L1, L2, L3, L4 = 1.5, 1.5, 1.5, 1.5
theta_rad = np.array([np.pi/4, -np.pi/4, -np.pi/2, 0])

# Screws
S = np.array([ [0,0,0,0],
               [0,0,0,0],
               [1,1,1,1],
               [0,0,0,0],
               [0, -L1, -(L1+L2), -(L1+L2+L3)],
               [0,0,0,0] ])
print('Space screws:\n', S)


num_joints = np.shape(S)[1] # determining the nubmer of joints
Js = np.zeros((6,num_joints)) # initializing the Jacobian array
Js[:,0] = S[:,0] # setting first column of J (J1) to S1
T = np.eye(4) # creating a starting transformation matrix
for i in range(1, num_joints):
    T = T @ ME4140.MatrixExp6(ME4140.VecTose3(S[:,i-1]*theta_rad[i-1]) )
    Js[:,i] = ME4140.Adjoint(T) @ S[:,i]

print('Space Jacobian:\n', Js)


#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Group Activity #2 (Psueodoinverse calculation for 2-dof)')
J = np.array([ [1, 3],
               [2, 4],
               [3, 3] ])

print_steps = True
if print_steps:
    print('J.T:\n', J.T)
    print('\nJ.T @ J:\n', J.T @ J)
    print('\n(J.T @ J)^-1:\n', np.linalg.inv(J.T @ J))
    
Jpinv = np.linalg.inv(J.T @ J) @ J.T
print('\nThe psueodoinverse of J is:\n', Jpinv)

print('\nThe psueodoinverse of J using numpy pinv is:\n', np.linalg.pinv(J))
#-----------------------------------------------------------------------#


#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions toGroup Activity #3 (Psueodoinverse calculation for 6-dof)')
J = np.array([ [0, 0, 0, 0.966, 0, 0.256],
               [0, 1, 1, 0, 1, 0],
               [1, 0, 0, -0.259, 0, -0.966],
               [0, -0.675, -1.488, 0, -1.138, 0],
               [0, 0, 0, 1.277, 0, 0.957],
               [0, 0.35, -0.463, 0, 0.685, 0]
               ])

print('\nThe inverse of J is:\n', np.linalg.pinv(J))

Jpinv = np.linalg.inv(J.T @ J) @ J.T
print('\nThe left psueodoinverse of J is:\n', Jpinv)

Jpinv = J.T @ np.linalg.inv(J @ J.T) 
print('\nThe right psueodoinverse of J is:\n', Jpinv)

print('\nThe inverse of J using the numpy function pinv:\n', np.linalg.pinv(J))
#-----------------------------------------------------------------------#


#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Group Activity #4 (Psueodoinverse calculation for 7-dof)')
S = np.array([ [0, 1,   0,  1,   0, 1,    0],
               [0, 0,   0,  0,   0, 0,    0],
               [1, 0,   1,  0,   1, 0,    1],
               [0, 0,   0,  0,   0, 0,    0],
               [0, 350, 0,  0,   0, 1170, 0],
               [0, 0,   0,  0,   0, 0,    0]
               ])
theta_deg = np.array([10,20,30,10,45,5,5])
theta_rad = np.deg2rad(theta_deg)
J = ME4140.JacobianSpace(S, theta_rad)
print('\nJacobian:\n', J)

m,n = np.shape(J)
print('\nm x n: {} x {}\n'.format(m,n))

if m == n:
    Jinv = np.linalg.inv(J)
    print('Calculating the inverse of J')
elif m > n: # LEFT inverse
    Jinv = np.linalg.inv(J.T @ J) @ J.T
    print('Calculating the LEFT Pseudo inverse of J')
elif m < n: # RIGHT inverse
    print('Calculating the RIGHT Pseudo inverse of J')
    #Jinv = J.T @ np.linalg.inv(J @ J.T)
    Jinv = J.T @ np.linalg.inv(J @ J.T)
    
print('\nThe inverse of J is:\n', Jinv)
print('\nThe inverse of J using the numpy function pinv:\n', np.linalg.pinv(J))
#-----------------------------------------------------------------------#