'''
Description: Python code for examples of singular matricies and rank and group activity in Lecture #24 (ME4140_L23_Inverse_Kinematics_pt1)
Author: Phil Deierling
Date: 04/04/2022
Version: 1.0
Log: 
04/04/2022: First submission
'''

import numpy as np
import sys
import scipy.constants as spc
import matplotlib.pyplot as plt


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
print('\nNumpy arctan2 cases')

xy = np.array( [ [1,1],  # Q1 (45 deg)
                 [-1,1], # Q2 (135 deg) 
                 [-1,-1],# Q3 (-135 deg)
                 [1,-1], # Q4 (-45 deg)
                 [0,1],  # Q1/Q2 line (90 deg)
                 [0,-1], # Q3/Q4 line (-90 deg)
                 [0,0]   # Origin
                    ])
                    
for x,y in xy:
    #print(x,y)  
    ang = np.arctan2(y,x)
    if x > 0:
        prt_str = 'a>0, '
    elif x < 0:
        prt_str = 'a<0, '
    else:
        prt_str = 'a=0, '
        
    if y > 0:
        prt_str += 'b>0 '
    elif y < 0:
        prt_str += 'b<0 '
    else:
        prt_str += 'b=0 '
        
    prt_str += 'case angle:'
    print(prt_str, ang)


'''
#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Example #1 (Planar 2R wrist up)')
L1 = 2.0
L2 = 1.5

thetaDeg = np.array([30, 30])
theta = np.radians(thetaDeg)

# Forward kinematics
x = L1*np.cos(theta[0]) + L2*np.cos(theta[0] + theta[1])
y = L1*np.sin(theta[0]) + L2*np.sin(theta[0] + theta[1])
X = np.array([x, y])
print('Forward kinematics position: ', X)

# Inverse kinematics
# Using the (x,y) from the forward kinematics we should be able to obtain 
# the same angles through inverse kinematics. 
b = np.sqrt(X[0]**2 + X[1]**2)
#print('b = ', b)

gamma = np.arctan2(X[1], X[0]) # y, x
alpha = np.arccos((-L2**2 + b**2 + L1**2)/(2*b*L1))
beta = np.arccos((L2**2 - b**2 + L1**2)/(2*L1*L2))
#print('Angles alpha, beta, and gamma: ', alpha, beta, gamma)

theta = np.array([gamma-alpha, np.pi-beta])
print('Joint angles to obtain the end-effector point ', X, 'are', theta, '(radians) or ', np.degrees(theta), '(degrees)')

# Forward check
x = L1*np.cos(theta[0]) + L2*np.cos(theta[0] + theta[1])
y = L1*np.sin(theta[0]) + L2*np.sin(theta[0] + theta[1])
X_check = np.array([x,y])
print('Forward kinematics position check: ', X_check)



#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Example #2 (Planar 2R wrist down)')
gamma = np.arctan2(X[1], X[0]) # y, x
alpha = np.arccos((-L2**2 + b**2 + L1**2)/(2*b*L1))
beta = np.arccos((L2**2 - b**2 + L1**2)/(2*L1*L2))
#print('Angles alpha, beta, and gamma: ', alpha, beta, gamma)

theta = np.array([alpha+gamma, beta-np.pi])
print('Joint angles to obtain the end-effector point ', X, 'are', theta, '(radians) or ', np.degrees(theta), '(degrees)')

# Forward check
x = L1*np.cos(theta[0]) + L2*np.cos(theta[0] + theta[1])
y = L1*np.sin(theta[0]) + L2*np.sin(theta[0] + theta[1])
X_check = np.array([x,y])
print('Forward kinematics position check: ', X_check)
'''