'''
Description: Various Functions created for ME 4140
Author: Mia Scoblic
Date: 02/03/2024
'''

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Description: Family of functions to calculate rotation matrices
# ----------------------------------------------------------------------------------------------------------------------

# Rotation matrix of some matrix around the space frame ( Rsa )
# Inputs: X hat vector and Y hat vector of some matrix
# Outputs: Rotation matrix in terms of space frame
def s_a(x_a, y_a):
    z_a = np.cross(x_a, y_a) # Calculates the third coordinate vector
    Rsa = np.column_stack((x_a, y_a, z_a))
    return Rsa

# Rotation matrix of space frame around some matrix ( Ras )
# Inputs: X hat vector and Y hat vector of some matrix
# Outputs: Transposes original matrix Rsa to get Ras
def a_s(x_a, y_a):
    z_a = np.cross(x_a, y_a)
    Ras = np.transpose(np.column_stack((x_a, y_a, z_a)))
    return Ras

# Calculates Rab given Ras and Rsb
# Inputs: Ras and Rsb
# Outputs: Rab
def a_b(Ras, Rsb):
    Rab = Ras @ Rsb
    return Rab
