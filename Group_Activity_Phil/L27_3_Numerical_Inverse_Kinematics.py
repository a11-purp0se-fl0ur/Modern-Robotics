# Number of decimals to round for printing
from matplotlib import pyplot as plt
from Functions.Mia_Functions import *
import modern_robotics as mr

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