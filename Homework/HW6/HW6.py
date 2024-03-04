from Functions.Mia_Functions import *

# Problem 1 ------------------------------------------------------------------------------------------------------------
print('\nProblem 1:')
# Given:
q = np.array([2, 4, 1]) # Point on the axis
s = np.array([np.pi/4, np.pi/6, np.pi/6]) # Axis of rotation
sHat = normalize(s) # Normalized to unit vector
h = 4 # Pitch
theta = np.pi/3 # Theta - unused

# Find:
# Screw Axis Sb
Sb_1 = parametersToScrew(sHat, q, h)
print('Screw Axis: ', Sb_1)

# Problem 2 ------------------------------------------------------------------------------------------------------------
print('\nProblem 2:')
# Given:
grav_2 = 10 #m/s^2
m_apple = 0.1 #kg
m_hand = 0.5 #kg
L1_2 = 0.1 #m
L2_2 = 0.15 #m

# Find:
# Force-Torque sensor in {f} frame
Ffh = np.array([0, -grav_2*m_hand, 0]) # Force on hand w.r.t sensor {f}
Rfh = np.array([L1_2, 0, 0]) # R-vector from sensor to force

Ffa = np.array([0, -grav_2*m_apple, 0]) # Force on apple w.r.t sensor {f}
Rfa = np.array([L1_2+L2_2, 0, 0]) # R-vector from sensor to force

Wfh = Wrench(Ffh, Rfh) # Wrench from hand
Wfa = Wrench(Ffa, Rfa) # Wrench from apple

W = Wfh + Wfa # Wrenches are additive, compile into total wrench

print('Total Wrench:', W)

# Problem 3 ------------------------------------------------------------------------------------------------------------
print('\nProblem 3:')
# Given:
L1_3 = 0.35
L2_3 = 0.41
L3_3 = 0.41
L4_3 = 0.136

fb = np.array([10, 5, 0])
rb = np.array([0,0,0])

Wb = Wrench(fb, rb)
print(Wb)

# Problem 4 ------------------------------------------------------------------------------------------------------------
print('\nProblem 4:')
# Given:
# Same problem as 3, just now express in the {s} frame

# Method 1: Redo problem from new frame
rsb = np.array([0, 0, L1_3 + L2_3 + L3_3 + L4_3])
Ws = Wrench(fb, rsb)
print('Wrench in {s}: ', Ws)

# Method 2: Use the answer from 3 and the adjoint of the transformation matrix
Rsb = np.eye(3)
psb = np.array([0, 0, L1_3 + L2_3 + L3_3 + L4_3])


Tsb = constructT(Rsb, psb)
Tbs = np.linalg.inv(Tsb)
Ad_Tbs = adjoint(Tbs)
Ws = Ad_Tbs.T @ Wb
print('Wrench in {s}: ', Ws)

