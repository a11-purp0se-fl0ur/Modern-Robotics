'''
Description: Python code for solving the example given in Lecture #19 (ME4140_L20_Manipulator_Jacobian)
Author: Phil Deierling
Date: 03/08/2022
Version: 1.0
Log: 
03/08/2022: First submission
'''

import numpy as np
import sys
import scipy.constants as spc

# Add the directory where the function for ME4140 are located
sys.path.append('/home/phil/Storage/Teaching/CourseResources/ME4140/Codes/Python')
import ME4140_Functions as ME4140


# Number of decimals to round for printing
dec = 3
np.set_printoptions(precision=3, suppress=True)

L1 = 1
L2 = 1
theta1 = 0
#theta2 = np.pi/4
#theta2 = 3*np.pi/4
#theta2 = 0
theta2 = np.pi
J = np.array([ [-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2), -L2*np.sin(theta1 + theta2)],
                [L1*np.cos(theta1) + L2*np.cos(theta1 + theta2), L2*np.cos(theta1 + theta2)] ])

print('Jacobian: \n', J)

#print('Inverse of J: \n', np.linalg.inv(J))

Rank_J = np.linalg.matrix_rank(J)
print('Rank of the Jacobian: ', Rank_J)

det_J = np.linalg.det(J)
if (det_J):
    print('Jacobian is NOT singular')
    print('Determinant is: ', det_J)
else:
    print('Jacobian is singular')
    print('Determinant is: ', det_J)



#print('Jacobian is singular ', Rank_J != J.shape[0])