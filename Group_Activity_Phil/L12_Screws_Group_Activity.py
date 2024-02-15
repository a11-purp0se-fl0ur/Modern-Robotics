'''
Description: Python code for solving the group activity in ME4140_L12_Screws
Author: Phil Deierling
Date: 02/18/2022
Version: 1.0
Log: 
02/18/2022: First submission
02/20/2023: Updated lecture number to 12
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
print('Solutions to Group Activity #1')
shat = 0.577*np.ones([3])
q = np.array([1,1,2])
h = 10

S = np.zeros(6) # initialize the screw axis as a row vector
S[:3] = shat # setting the first 3 rows equal to shat
S[3:] = np.cross(-shat, q) + h*shat # setting the last 3 rows equal to -shat x q + h*shat
print('\nScrew axis S = \n', S)

S_byfunc = ME4140.ScrewToAxis(q, shat, h) # using a function
print('\nScrew axis (using function) S = \n', S_byfunc)
print('#-----------------------------------------------------------------------#')


#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('Solutions to Group Activity #2')
V = np.array([1.091, 2.182, 4.365, 2.183, -3.274, 1.091]).reshape(-1,1)
theta_dot = np.linalg.norm(V[:3]) # since w !=0, theta_dot is ||w||
S = V/theta_dot # screw axis is the twist normalized by theta_dot
print('\nScrew axis S = \n', S)

Sw = S[:3] # rotation components Sw
Sv = S[3:] # linear components Sv
h = np.dot(Sw.T,Sv) # outputs as an array
h = h[0,0]
print('Screw pitch h = {:.3f}\n'.format(h))

shat = Sw
print('\nshat = \n', shat)

Svr = Sv - h*shat
print('Shape of Svr = ', np.shape(Svr))
print('Shape of shat = ', np.shape(shat))

q = np.cross(shat.T, Svr.T)
print('\nPoint on the axis q = \n', q)
print('#-----------------------------------------------------------------------#')


#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('Solutions to Group Activity #3')
omega = np.array([1,2,1])
q = np.array([1,1,2])
h = 10

theta_dot = np.linalg.norm(omega) # since w !=0, theta_dot = ||w||
print('\nTheta dot = \n', theta_dot)

shat = omega/theta_dot # since w !=0, shat = w/theta_dot
S = ME4140.ScrewToAxis(q, shat, h)
print('\nScrew axis S = \n', S)

V = S*theta_dot
print('\nTwist V = \n', V)

