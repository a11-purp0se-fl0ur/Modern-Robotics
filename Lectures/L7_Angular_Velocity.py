'''
Description: Python code for solving the group activity in ME4140_L07_Angular_Velocity
Author: Phil Deierling
Date: 01/31/2024
Version: 1.0
Log: 
01/31/2024: First submission
'''

import numpy as np
import ME4140_Functions as ME4140 # You will need to create your own functions file to run this.                

# Original orientation of the {b} frame
Rsb = np.eye(3)
print("Original orientation of the {b} frame:\n", Rsb, '\n')

# Rotation operator
R = ME4140.Rot('z', 30, 'deg') @ ME4140.Rot('x', 40, 'deg')
print('Rotation operator:\n', R, '\n')

# New orientation of {b'} frame
Rsb_prime = R @ Rsb
print("New orientation of the {b'} frame:\n", np.round(Rsb_prime,3), '\n')

# Angular velocity in the {s} frame
ws = np.array([3,3,2])

# Angular velocity in the {b'} frame
wb_prime = Rsb_prime.T @ ws
print("Angular velocity in the {b'} frame (row vector):\n", np.round(wb_prime,3), '\n')

wb_prime = wb_prime.reshape(-1,1)
print("Angular velocity in the {b'} frame (column vector):\n", np.round(wb_prime,3), '\n')