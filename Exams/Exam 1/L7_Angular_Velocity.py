from Functions.Mia_Functions import *

dec = 3
np.set_printoptions(precision=3, suppress=True)

# Group Activity
# Given:
Rsb_prime = Rot('z', 30, 'deg') @ Rot('x', 40, 'deg')
Ws = np.array([3, 3, 2])

# Find:
# (1) Rsb'
print('Rsb_prime:\n', Rsb_prime)

# (2) Wb'
Rbs_prime = np.transpose(Rsb_prime)
Wb_prime = Rbs_prime @ Ws
print('Wb_prime:\n', Wb_prime)
