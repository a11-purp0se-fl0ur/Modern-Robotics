from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Group Activity

# Given:
Rsb = Rot('y', np.pi/4, 'rad')
psb = np.array([-1, -2, 0])
Vb = np.array([1, 2, 1, 0, 0, 0])

# Find:
# (1) Twist is {s} frame
Tsb = constructT(Rsb, psb)

Ad_Tsb = adjoint(Tsb)
#Ad_Tsb_skew = skew(Ad_Tsb)

Vs = Ad_Tsb @ Vb

print('Vs:\n', Vs)