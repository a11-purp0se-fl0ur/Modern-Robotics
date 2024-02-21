'''
Description: Python code for solving the group activity in ME4140_L13_Wrenches
Author: Phil Deierling
Date: 03/12/2021
Version: 1.02
Log: 
03/12/2021: First submission
02/21/2022: Added a second example with different numbers so students can run a second case. 
02/20/2023: Changed lecture number to 13 (14 previous) 
'''

import numpy as np
import sys

# Add the directory where the function for ME4140 are located
#sys.path.append('/home/phil/Storage/Teaching/CourseResources/ME4140/Codes/Python')
import ME4140_Functions as ME4140


# Number of decimals to round for printing
dec = 3
np.set_printoptions(precision=3, suppress=True)


#-----------------------------------------------------------------------#
print('\n#-----------------------------------------------------------------------#')
print('Solutions to Class Activity')


# Given
pa = np.array( [-3, 2, 4] )
rb = np.array( [2, 3, -4] )
fb = np.array( [-1, 2, 2] ) # force in {b}
Rab = ME4140.Rot('z', 30, 'Deg') @ ME4140.Rot('x', 30, 'Deg') 
print('\nRab: \n', Rab)

# Solution
# moment in {b}
mb = np.cross(rb, fb)
print('\nMoment in {b} mb: \n', mb)

# wrench in {b}
Fb = np.concatenate((mb, fb), axis=0) # bringing the moments and forces into a single array
print('\nWrench in {b} Fb: \n', Fb)

# transformation matrix
Tab = ME4140.RpToTrans(Rab, pa)
print('\nTab: \n', Tab)

Tba = np.linalg.inv(Tab)
print('\nTba: \n', Tba)

# Adjoint of Tba
Ad_Tba = ME4140.Adjoint(Tba)
print('\n[Adj Tba]: \n', Ad_Tba)

# wrench in {a}
Fa = Ad_Tba.T @ Fb
print('\nWrench in {a} Fa: \n', Fa)
print('\n#-----------------------------------------------------------------------#')


print('\n#-----------------------------------------------------------------------#')
print('Solutions to Class Activity (numbers check)')

# Given
pa = np.array( [3, 1, 5] )
rb = np.array( [1, 7, -1] )
fb = np.array( [2, -3, -4] )    # force in {b}
Rab = np.matmul( ME4140.Rot('z', 60, 'Deg'), ME4140.Rot('x', 45, 'Deg') )
print('\nRab: \n', Rab)

# Solution
# moment in {b}
mb = np.cross(rb, fb)
print('\nMoment in {b} mb: \n', mb)

# wrench in {b}
Fb = np.concatenate((mb, fb), axis=0)
print('\nWrench in {b} Fb: \n', Fb)

# transformation matrix
Tab = ME4140.RpToTrans(Rab, pa)
print('\nTab: \n', Tab)

Tba = np.linalg.inv(Tab)
print('\nTba: \n', Tba)

# Adjoint of Tba
Ad_Tba = ME4140.Adjoint(Tba)
print('\n[Adj Tba]: \n', Ad_Tba)

# wrench in {a}
Fa = np.matmul(Ad_Tba.T, Fb)
print('\nWrench in {a} Fa: \n', Fa)
print('\n#-----------------------------------------------------------------------#')
