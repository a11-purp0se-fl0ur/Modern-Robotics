'''
Description: Python code examples for review of matrix algebra.
Author: Phil Deierling
Date: 01/21/2022
Version: 1.0
Log: 
01/21/2022: First submission
01/19/2023: Added call to external functions file to demonstrate how to call functions from another file
'''




# Basic Python variables
a = 1 #integer
b = 1.0 #float
c = 'some string'
d = "another string"
A = 2

# Printing and variable types
print('The variable a is', a, 'and is an', type(a))
print('The variable b is', b, 'and is a', type(b))
print('These are the values of a %s and b %s.' %(a,b))
print('The variable A is', A, 'and is an', type(A)) 

str1 = "Jill,"
str2 = "Jack."
print("Hello {} hello {}".format(str1, str2)) # python3 specific


# How to import modules
import numpy as np # now anytime we want to use numpy we can just call np instead of numpy

# Vectors
e = np.array( [0,0,1] ) #row vector
print('Vector e:\n', e)
print('Shape of vector e:\n', np.shape(e))

# Matricies
B = np.zeros([4,3]) #4x3 matrix of all zeros
print('\nMatrix B:\n', B)


I3 = np.identity(3)
print('\nIdenity matrix (3x3):\n', I3)

C = np.array([ [1,2,3], [4,5,6], [7,8,9] ] )
print('\nMatrix C:\n', C)

D = np.array([ [1,2,3,4], 
               [5,6,7,8], 
               [9, 10,11, 12], 
               [13, 14, 15, 16] ] )
print('\nMatrix D:\n', D)


# Matrix multiplication
E = np.random.rand(4,4)
F = np.matmul(D,E)
print('\nThe result of matrix D multiplied with E is:\n', F)

# Matrix multiplication (another way)
H = D @ E
print('\nThe result of matrix D multiplied with E (other method) is:\n', H)

# Rounding the result
J = np.round(H,1)
print('\nThe rounded result of matrix H is:\n', J)


# Internal function example
def addMats(A, B):
    C = A + B
    return C


K = np.random.rand(3,3)
L = np.random.rand(3,3)
KplusL = addMats(K,L)
print('\nMatrix K + L:\n',KplusL)


# Transpose
G = np.transpose(F)
print('\nTranspose of G is:\n', G)



# Matrix, vector multiplication
D = np.array([ [1,2,3,4], 
               [5,6,7,8], 
               [9, 10,11, 12], 
               [13, 14, 15, 16] ] ) # 4x4 matrix

b = np.array([0,0,1,2]) # 1x4 (row vector)
print('\nRow vector b:\n', b)
print('Shape of vector b:\n', np.shape(b))

c = np.transpose(b)
print('\nTranspose of b is:\n', c)
print('Shape of vector b after np.transpose(b):\n', np.shape(c))


import ME4140_L02_Functions_Spring_2023 as fn # Note, the file name here is ME4140_L02_Functions.py and needs to match

# compare the shape of the two row vectors
fn.checkShape(b, c)

# Instead for a row vector we need to do something a little different
d = b.reshape(-1,1) # turn the row vector into a column vector (this is because the transpose won't work for a row or column vector)
print('\nShape of vector d:\n', np.shape(d))
fn.checkShape(b, d)
    
# Multiply a 4x4 matrix with a 4x1 vector (we expect to get a 4x1 vector out)    
print('\nResult of D*b (method 1):\n', np.dot(D,d))
print('\nResult of D*b (method 2):\n', D @ d)



# Inverse
H = np.array([[6, 1, 1],
              [4, -2, 5],
              [2, 8, 7]])
print('\nInverse of H is:\n', np.linalg.inv(H))
print('\nMy inverse of H (from function) is:\n', fn.myInv(H))



