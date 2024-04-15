from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')

# Given
Tsd = np.array([[0.707, -0.696, -0.123, -127.5], [0.707, 0.696, 0.123, 127.5], [0, -0.174, 0.985, 190], [0, 0, 0, 1]])
L1 = L2 = L3 = 100
b = 50
A = 350
B = 675
C = 1150
D = 41
E = 1200
F = 240

# Find the joint angles to achieve Tsd

# Home Configuration
Rsb = np.eye(3)
p = np.array([A+E+F, 0, B+C-D])
M = constructT(Rsb, p)
print('\nHome:\n', M)

# Screws in the {s} frame
h = 0
sHat1 = np.array([0, 0, 1])
sHat235 = np.array([0, 1, 0])
sHat46 = np.array([1, 0, 0])
q1 = np.array([0, 0, 0])
q2 = np.array([A, 0, B])
q3 = np.array([A, 0, B+C])
q4 = np.array([A+E, 0, B+C-D])
q5 = np.array([A+E, 0, B+C-D])
q6 = np.array([A+E+F, 0, B+C-D])

S1 = parametersToScrew(sHat1, q1, h)
S2 = parametersToScrew(sHat235, q2, h)
S3 = parametersToScrew(sHat235, q3, h)
S4 = parametersToScrew(sHat46, q4, h)
S5 = parametersToScrew(sHat235, q5, h)
S6 = parametersToScrew(sHat46, q6, h)
Ss = np.column_stack((S1, S2, S3, S4, S5, S6))
print('\nScrews in {s}:\n', Ss)

# Screws in {ee} frame
Sb = adjoint(np.linalg.inv(M)) @ Ss
print('\nScrews in {ee}:\n', Sb)

# Initial Guess
theta_deg = np.array([10, 10, 10, 10, 10, 10])
theta_rad = np.radians(theta_deg)

# Initialization
epsilon_w = 1e-3 # rotational error, rad
epsilon_v = 1e-3 # translational error, m
it = 0
itmax = 100
ew = 1e6
ev = 1e6
frame = 'space'

# Start of algorithm
print('\nSTART OF ALGORITHM')
print('iter\t theta1 (deg)\ttheta2 (deg)\t x\t y\t wz\t vx\t vy\t ew\t\t ev')
while (ew > epsilon_w or ev > epsilon_v) and it <= itmax:

    if frame == 'space':
        # Configuration at current theta
        Tsb = PoE_Space(theta_rad, M, Ss)

        Tbs = np.linalg.inv(Tsb)
        Tbd = Tbs @ Tsd

        Rbd, pbd = deconstructT(Tbd)

        # Body twist needed to move from {b} to {d}
        Vb = skew(Matrix_Logarithm_Rotations(Rbd))

        # Body twist in the space frame (space twist)
        Vs = adjoint(Tsb) @ Vb

        Js = SpaceJacobian(Ss, theta_rad)
        Jinv = np.linalg.pinv(Js)

        V = Vs

    else:
        # compute = False
        print('Please choose an appropriate frame (body or space) for the calculation.')
        break

    # error calculations
    ew = np.linalg.norm([V[0], V[1], V[2]])
    ev = np.linalg.norm([V[3], V[4], V[5]])

    theta_rad1 = theta_rad + Jinv @ V

    # End-effector coordinates
    x, y = Tsb[0:2, -1]

    print('{:d}\t {:.5f}\t{:.5f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3e}\t {:.3e}'.format(it, np.rad2deg(theta_rad[0]), np.rad2deg(theta_rad[1]),
                                                                                                        x, y, V[2], V[3], V[4], ew, ev))

    it += 1
    theta_rad = theta_rad1