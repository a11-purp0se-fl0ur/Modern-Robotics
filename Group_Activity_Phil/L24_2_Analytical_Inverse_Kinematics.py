'''
Description: Python code for examples of singular matricies and rank and group activity in Lecture #24 (ME4140_L24_Inverse_Kinematics_pt2)
Author: Phil Deierling
Date: 04/14/2023
Version: 1.0
Log: 
04/14/2023: First submission
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
print('#-----------------------------------------------------------------------#')
run=True
L1 = 10
L2 = 10
L3 = 10

arm_case = int(input('Input a case (1-4) to evaluate.'))
print('Arm case:', arm_case)
print(type(arm_case))
#arm_case = 5
if arm_case == 1:
    x = 15
    y = 20
    phiDeg = 85
elif arm_case == 2:
    x = 18.5
    y = 22.057371
    phiDeg = 70
elif arm_case == 3:
    x = L1+L2
    y = 0
    phiDeg = 0
elif arm_case == 4:
    x = L1
    y = L2
    phiDeg = 30
else:
    print("Please choose a valid case number (1-4). Try again.")
    run=False


if run:
    print('\nSolutions to Example #3 (Planar 3R)')
    X = np.array([x, y])
    phi = np.radians(phiDeg)
    print('Forward kinematics position ', X, 'and orientation (degrees)', phiDeg)


    # Inverse kinematics
    # Using the (x,y) from the forward kinematics we should be able to obtain 
    # the same angles through inverse kinematics. 
    X = np.array([x, y])
    c = np.sqrt(X[0]**2 + X[1]**2)

    # Finding the w point (point on the wrist)
    W = np.array([ X[0] - L3*np.cos(phi), X[1] - L3*np.sin(phi) ])
    b = np.sqrt(W[0]**2 + W[1]**2)
    print('Wrist point: ', W)

    if (L1+L2) > b:

        # Solving the J1/J2 Law of cosines
        delta = np.arctan2(W[1], W[0])
        alpha = np.arccos((-L2**2 + b**2 + L1**2)/(2*b*L1))
        beta = np.arccos((L2**2 - b**2 + L1**2)/(2*L2*L1))
        print('Angles delta, alpha, and beta: ', delta, alpha, beta)

        # Finding the joint angles (solution 1)
        theta1a = delta-alpha
        theta2a = np.pi-beta
        theta3a = phi - theta1a - theta2a
        thetaA = np.array([theta1a, theta2a, theta3a])
        print('Joint angles to obtain the end-effector point', X, 'with orientation', phiDeg, 'are', thetaA, '(radians) or', np.degrees(thetaA), '(degrees)')

        # Finding the joint angles (solution 2)
        theta1b = delta + alpha
        theta2b = beta - np.pi
        theta3b = phi - theta1b - theta2b
        thetaB = np.array([theta1b, theta2b, theta3b])

        print('Alternative joint angles to obtain the end-effector point', X, 'with orientation', phiDeg, 'are', thetaB, '(radians) or', np.degrees(thetaB), '(degrees)')

        # Array to hold points for plotting
        pnts1 = np.zeros([4,2]) # joint solution 1
        # Link 1
        pnts1[1,0] = L1*np.cos(thetaA[0])
        pnts1[1,1] = L1*np.sin(thetaA[0])
        # Link 2
        pnts1[2,:] = W
        # end-effector
        pnts1[3,:] = X
        #plt.scatter(pnts1[:,0], pnts1[:,1])
        #plt.plot(pnts1[:,0], pnts1[:,1], label = 'Orientation 1')

        # Array to hold points for plotting
        pnts2 = np.zeros([3,2]) # joint solution 2
        # Link 1
        pnts2[1,0] = L1*np.cos(thetaB[0])
        pnts2[1,1] = L1*np.sin(thetaB[0])
        # Link 2
        pnts2[2,:] = W

        # plot the solutions and workspace
        fig, axs = plt.subplots(1, 1)

        axs.scatter(pnts1[:,0], pnts1[:,1])
        axs.plot(pnts1[:,0], pnts1[:,1], label = 'Orientation 1')

        axs.scatter(pnts2[:,0], pnts2[:,1])
        axs.plot(pnts2[:,0], pnts2[:,1], label = 'Orientation 2') 

        R = L1+L2+L3
        circX = []
        circY = []
        angs = np.linspace(0, 2*np.pi,201)
        print(angs)
        for ang in angs:
            circX.append(R*np.cos(ang))
            circY.append(R*np.sin(ang))
        axs.plot(circX, circY) 

        axs.set_aspect('equal', adjustable='datalim')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Forward kinematics check
        x = L1*np.cos(thetaA[0]) + L2*np.cos(thetaA[0] + thetaA[1]) + L3*np.cos(thetaA[0] + thetaA[1] + thetaA[2])
        y = L1*np.sin(thetaA[0]) + L2*np.sin(thetaA[0] + thetaA[1]) + L3*np.sin(thetaA[0] + thetaA[1] + thetaA[2])
        phi = np.sum(thetaA)
        X_check = np.array([x,y, np.degrees(phi)])
        print('Forward kinematics position and orientation (degrees) check: ', X_check)


    else:
        print('Error: End-effector is outside the workspace!')
            
            
            
            
            
            
            
