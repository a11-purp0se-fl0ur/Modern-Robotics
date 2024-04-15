'''
Description: Python code for solving examples in Lecture #26 (ME4140_L26_Numerical_Inverse_Kinematics_pt1)
Author: Phil Deierling
Date: 04/15/2022
Version: 1.0
Log: 
04/15/2022: First submission
04/24/2023: Corrected standard error to be the difference between function evaluations, not between the points.
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
print('\nSolutions to Example #1 (Newton-Raphson 1D)')
def fx(x):
    fx = x**3 - x - 1 # function
    return fx

def fpx(x):
    fpx = 3*x**2 - 1  # derivative
    return fpx

x0 = 1 # initial guess
epsilon = 1e-5
error = 1e6
iter = 1
print('iter\t xi\t\tfx \t\tfpx \t\terror')
while error > epsilon and iter < 100:
    x1 = x0 - fx(x0)/fpx(x0)
    error = np.abs(x1 - x0)
    print('{:d} \t {:.7f} \t{:.7f} \t{:.7f} \t{:.7f}'.format(iter,x1, fx(x0), fpx(x0), error))
    iter += 1
    x0 = x1

iter -= 1
print('\nSolution:', x1, ' at iteration: ', iter, ' with error: ', error)

# Plot the function to verify the root
x = np.linspace(0,2)
y = fx(x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()
#-----------------------------------------------------------------------#


#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Example #2 (Newton-Raphson 1D, multiple solutions)')
def fx(x):
    fx3 = -x**4 + 8*x**2 + 4
    return fx3

def fpx(x):
    fpx3 = -4*x**3 + 16*x
    return fpx3

x0 = -3 # initial guess
epsilon = 1e-5
error = 1e6
iter = 1
print('iter\t xi\t\tfx \t\tfpx \t\terror')
while error > epsilon and iter < 100:
    x1 = x0 - fx(x0)/fpx(x0)
    error = np.abs(x1 - x0)
    print('{:d} \t {:.7f} \t{:.7f} \t{:.7f} \t{:.7f}'.format(iter,x1, fx(x0), fpx(x0), error))
    iter += 1
    x0 = x1

iter -= 1
print('\nSolution:', x1, ' at iteration: ', iter, ' with error: ', error)

# Plot the function to verify the root
x = np.linspace(-3,3)
y = fx(x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()
#-----------------------------------------------------------------------#


#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Example #3 (Newton-Raphson 2D)')
def fx(X):
    x,y = X
    fx1 = x**2 + y**2 - 4
    fx2 = 4*x**2 - y**2 - 4
    return np.array([fx1, fx2])

def fpx(X):
    x,y = X
    df1dx = 2*x
    df1dy = 2*y
    df2dx = 8*x
    df2dy = -2*y
    return np.array([ [df1dx, df1dy], 
                      [df2dx, df2dy] ])

X0 = np.array([1,1]) # initial guess
epsilon = 1e-5
error = 1e6
iter = 1
print('iter\t xi\t\tyi \t\tf(x1) \t\tf(y1) \t\tf(x0) \t\tf(y0) \t\terror')
while error > epsilon and iter < 100:
    X1 = X0 - np.linalg.inv(fpx(X0)) @ fx(X0)
    f1 = fx(X1)
    f2 = fx(X0)
    error = np.linalg.norm(X1 - X0)
    print('{:d} \t {:.7f} \t{:.7f} \t{:.7f} \t{:.7f} \t{:.7f} \t{:.7f} \t{:.7f}'.format(iter,X1[0], X1[1], f1[0], f1[1], f2[0], f2[1], error))
    iter += 1
    X0 = X1

iter -= 1
print('\nSolution:', X1, ' at iteration: ', iter, ' with error: ', error)
#-----------------------------------------------------------------------#

