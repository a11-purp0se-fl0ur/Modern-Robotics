from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Group Activity #1
# Given:
s_Hat1 = np.array([.577, .577, .577])
q1 = np.array([1, 1, 2])
h1 = 10

# Find:
# (1) Screw Axis S
S1 = parametersToScrew(s_Hat1, q1, h1)
print("S: \n", S1)

# Group Activity #2
# Given:
V2 = np.array([1.091, 2.182, 4.365, 2.183, -3.274, 1.091])

# Find:
# (1) The Screw S
S2 = twistToScrew(V2)
print("S: \n", S2)

# (2) Screw parameters
h2, s_Hat2, q2 = screwToParameters(S2)
print("Pitch:\n", h2)
print("Axis of Rotation: \n", s_Hat2)
print("Point on the Axis: \n", q2)

# Group Activity #3
# Given:
w3 = np.array([1, 2, 1])
q3 = np.array([1, 1, 2])
h3 = 10

# Find:
# (1) Screw axis S
s3_Hat = normalize(w3)

S3 = parametersToScrew(s3_Hat, q3, h3)
print("S: \n", S3)

# (2) Twist V
thetadot = np.linalg.norm(w3)
V3 = S3*thetadot

print("Twist: \n", V3)