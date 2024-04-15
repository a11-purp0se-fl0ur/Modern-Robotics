from Functions.Mia_Functions import *
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Test 1
print('\nProblem 1:\n')
A = np.array([[1, -2, 3],[2, -3, 5],[1, 1, 0]])
checkA = singularity(A)

# Test 2
print('\nProblem 2:\n')
L1 = 1
L2 = 1
theta1 = 0
theta2 = np.pi/4
#theta2 = 3*np.pi/4
#theta2 = 0
#theta2 = np.pi
J = np.array([ [-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2), -L2*np.sin(theta1
+ theta2)],[L1*np.cos(theta1) + L2*np.cos(theta1 + theta2), L2*np.cos(theta1 + theta2)] ])

checkJ = singularity(J)

