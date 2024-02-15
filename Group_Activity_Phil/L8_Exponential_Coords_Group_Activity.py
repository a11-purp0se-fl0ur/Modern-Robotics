'''
Description: Python code for solving the group activity in ME4140_L08_Exponential_Coordinates 
Author: Phil Deierling
Date: 02/17/2021
Version: 1.0
Log: 
02/17/2021: First submission
02/17/2023: Changed lecture number (10->8) and added Rodrigues formula before the function call version. 
'''

import numpy as np
from Functions.Phil_Functions import * # You will need to create your own functions file to run this.
                     
wHat = np.array([0,0.866,0.5])
thetad = 30
theta_rad = np.radians(thetad)
print('Angle in radians: ', theta_rad)


expCoords = wHat*theta_rad
print('Exponential coordinates: \n', expCoords)


wso3 = ME4140.VecToso3(wHat)
print('\nwso3 = \n', wso3)

# Explicitly with Rodrigues formula
Rsb = np.identity(3) + np.sin(theta_rad)*wso3 + (1-np.cos(theta_rad))*(wso3 @ wso3)
print('\nRotation matrix (Rodrigues): \n', Rsb)


# Using my function
#Rsb = ME4140.MatrixExp3(wso3, theta_rad)
Rsb = ME4140.MatrixExp3(wso3*theta_rad)
print('\nRotation matrix (function): \n', Rsb)