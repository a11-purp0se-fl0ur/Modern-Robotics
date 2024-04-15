'''
Description: Python code for solving examples in Lecture #27 (ME4140_L26_Numerical_Inverse_Kinematics_pt3)
Author: Phil Deierling
Date: 04/20/2022
Version: 1.0
Log: 
04/20/2022: First submission
04/21/2023: Added 
04/05/2024: Separeated pt2 into pt2 & 3 
'''

import numpy as np
import sys
import scipy.constants as spc
import matplotlib.pyplot as plt


# Add the directory where the function for ME4140 are located
try:
    sys.path.append('/home/phil/Documents/Teaching/CourseResources/ME4140/Codes/Python')
except:
    pass
import ME4140_Functions as ME4140


# Number of decimals to round for printing
dec = 3
np.set_printoptions(precision=3, suppress=True)


#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Group Activity #1 (2R robot inverse kinematics coordinate based)')
# This method uses the Jacobian using trigonometry
L1 = 1
L2 = 1

# Desired coordinate vector xd
thetaDeg = np.array([30, 90])
theta = np.radians(thetaDeg)

# Forward kinematics
x = L1*np.cos(theta[0]) + L2*np.cos(theta[0] + theta[1]) # using just to establish a point for verification
y = L1*np.sin(theta[0]) + L2*np.sin(theta[0] + theta[1])
Xd = np.array([x, y])
print('Desired coordinate position:\n', Xd)

# Initial guess
theta_deg0 = np.array([10,10])
theta_rad0 = np.deg2rad(theta_deg0)

# Initialization
epsilon = 1e-3
error = 1E6
it = 0
itmax = 50

# Start of algorithm
print('\nSTART OF ALGORITHM')
print_header = ('iter', 'theta (deg)', '(x,y)', 'error')
print('{:5} {:>5} {:>13} {:>25}'.format(*print_header) )

while it <= itmax:
    
    # Current Jacobian and inverse evaluated at theta_rad0
    Jbody = np.array([ [-L1*np.sin(theta_rad0[0]) - L2*np.sin(theta_rad0[0]+theta_rad0[1]), -L2*np.sin(theta_rad0[0]+theta_rad0[1])],
                  [ L1*np.cos(theta_rad0[0]) + L2*np.cos(theta_rad0[0]+theta_rad0[1]), L2*np.cos(theta_rad0[0]+theta_rad0[1])]])

    Jbody_inv = np.linalg.pinv(Jbody)
    
    # Current coordinates evaluated at theta_rad0
    x = L1*np.cos(theta_rad0[0]) + L2*np.cos(theta_rad0[0] + theta_rad0[1]) # using just to establish a point for verification
    y = L1*np.sin(theta_rad0[0]) + L2*np.sin(theta_rad0[0] + theta_rad0[1])
    X0 = np.array([x, y])
    
    # Difference between desired and current coordinates
    diff = Xd - X0

    # Calculation of next iteration angles
    error = np.linalg.norm(diff)
    
    # print iteration results
    ang_print = np.array2string(np.rad2deg(theta_rad0), formatter={'float_kind':'{:<8.2f}'.format})
    xyz_print = np.array2string(np.array([x,y]), formatter={'float_kind':'{:<10.2e}'.format})
    errors_print = np.array2string(np.array([error]), formatter={'float_kind':'{:<6.2e}'.format})
    print('{:<5d} {} {} {}'.format(it,  ang_print, xyz_print, errors_print) )
    
    # If the error is <= to the tolerance, then break (exit) the while loop
    if error <= epsilon:
        break
    
    # Update variables
    theta_rad1 = theta_rad0 + Jbody_inv @ diff
    theta_rad0 = theta_rad1
    it += 1
    
    # Check if any of the new angles are outside of the (0, 2*pi) range, if so bring them back in
    # For example: theta = 480 deg becomes theta = 120 deg (480-360)
    for i in range(len(theta_rad0)):
        if theta_rad0[i] < 0:
            while theta_rad0[i] < 0:
                theta_rad0[i] += 2*np.pi
        elif theta_rad0[i] > 2*np.pi:
            while theta_rad0[i] > 2*np.pi:
                theta_rad0[i] -= 2*np.pi
   

print('\n\nDesired coordinate position:   ', Xd)
print('Calculated coordinate position:', X0)

# Plotting the results
x, y = L1*np.cos(theta_rad0[0]), L1*np.sin(theta_rad0[0])
link1 = np.array([ [0, 0],
                   [x,y] ])

x = L1*np.cos(theta_rad0[0]) + L2*np.cos(theta_rad0[0] + theta_rad0[1]) # using just to establish a point for verification
y = L1*np.sin(theta_rad0[0]) + L2*np.sin(theta_rad0[0] + theta_rad0[1])
link2 = np.array([ link1[1,:], [x,y] ])

fig, axs = plt.subplots(1, 1)
axs.scatter(link1[:,0], link1[:,1], color='red')
axs.scatter(link2[:,0], link2[:,1], color='red')
axs.plot(link1[:,0], link1[:,1], color='black')
axs.plot(link2[:,0], link2[:,1], color='black')
axs.set_aspect('equal', adjustable='datalim')
plt.legend()
plt.grid(True)
plt.show()
        

#-----------------------------------------------------------------------#
print('#-----------------------------------------------------------------------#')
print('\nSolutions to Group Activity #1 (2R robot inverse kinematics configuration based)')
L1 = 1
L2 = 1

# Home matrix
Rsb = np.eye(3)
p = np.array([L1+L2, 0, 0])
M = ME4140.RpToTrans(Rsb,p)
print('\nHome:\n', M)

# Desired configuration
Tsd = np.array([ [-0.5, -0.866, 0, 0.366],
                 [0.866, -0.5,   0, 1.366],
                 [0,     0,     1, 0],
                 [0,     0,     0, 1] ])
print('\nDesired configuration:\n', Tsd) # this should corespond to theta=[30,90] degrees

# Robot screws in the space frame
S = np.array([ [0, 0],
               [0, 0],
               [1, 1],
               [0, 0],
               [0, -L1],
               [0, 0] ])

# Robot screws in the body frame
B = ME4140.Adjoint(np.linalg.inv(M)) @ S

# Initial guess
theta_deg0 = np.array([10,10])
theta_rad0 = np.deg2rad(theta_deg0)
    
# Initialization
epsilon_w = 1e-3 # rotational error, rad
epsilon_v = 1e-3 # translational error, m
it = 0
itmax = 100
ew = 1e6
ev = 1e6
frame = 'body'
#frame = 'space'

# Start of algorithm
print('\nSTART OF ALGORITHM')
print('iter\t theta1 (deg)\ttheta2 (deg)\t x\t y\t wz\t vx\t vy\t ew\t\t ev')
while (ew > epsilon_w or ev > epsilon_v) and it <= itmax:
    
    if frame == 'space':
        # Configuration at current theta
        Tsb = ME4140.FKinSpace(M, S, theta_rad0)
        
        Tbs = np.linalg.inv(Tsb)
        Tbd = Tbs @ Tsd
    
        # Body twist needed to move from {b} to {d}
        Vb = ME4140.se3ToVec(ME4140.MatrixLog6(Tbd))
        
        # Body twist in the space frame (space twist)
        Vs = ME4140.Adjoint(Tsb) @ Vb

        Js = ME4140.JacobianSpace(S, theta_rad0)
        Jinv = np.linalg.pinv(Js)

        V = Vs
    
    elif frame == 'body':
    
        Tsb = ME4140.FKinBody(M, B, theta_rad0)
        J = ME4140.JacobianBody(B, theta_rad0)
    
        Tbs = np.linalg.inv(Tsb)
        Jinv = np.linalg.pinv(J)
    
        Tbd = Tbs @ Tsd
    
        Vb_bracket = ME4140.MatrixLog6(Tbd)
        Vb = ME4140.se3ToVec(Vb_bracket)
        V = Vb
    
    else:
        #compute = False
        print('Please choose an appropriate frame (body or space) for the calculation.')
        break
        
    # error calculations
    ew = np.linalg.norm([V[0], V[1], V[2]])
    ev = np.linalg.norm([V[3], V[4], V[5]])

    theta_rad1 = theta_rad0 + Jinv @ V
    
    # End-effector coordinates
    x,y = Tsb[0:2,-1]

    print('{:d}\t {:.5f}\t{:.5f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3e}\t {:.3e}'.format(it, np.rad2deg(theta_rad0[0]), np.rad2deg(theta_rad0[1]),
                                                                     x, y, V[2], V[3], V[4], ew, ev ) )
    
    it += 1
    theta_rad0 = theta_rad1


print('\n############# Verification #############')
print('Desired end-effector configuration Tsd:\n', Tsd)
if frame == 'space':
    Tsd_verify = ME4140.FKinSpace(M, S, theta_rad0)
    print('\nComputed end-effector configuration:\n', Tsd_verify)
elif frame == 'body':
    Tsd_verify = ME4140.FKinBody(M, B, theta_rad0)
    print('\nComputed end-effector configuration:\n', Tsd_verify)


# Plotting the results
x, y = L1*np.cos(theta_rad0[0]), L1*np.sin(theta_rad0[0])
link1 = np.array([ [0, 0],
                   [x,y] ])

x = L1*np.cos(theta_rad0[0]) + L2*np.cos(theta_rad0[0] + theta_rad0[1]) # using just to establish a point for verification
y = L1*np.sin(theta_rad0[0]) + L2*np.sin(theta_rad0[0] + theta_rad0[1])
link2 = np.array([ link1[1,:], [x,y] ])

fig, axs = plt.subplots(1, 1)
axs.scatter(link1[:,0], link1[:,1], color='red')
axs.scatter(link2[:,0], link2[:,1], color='red')
axs.plot(link1[:,0], link1[:,1], color='black')
axs.plot(link2[:,0], link2[:,1], color='black')
axs.set_aspect('equal', adjustable='datalim')
plt.legend()
plt.grid(True)
plt.show()