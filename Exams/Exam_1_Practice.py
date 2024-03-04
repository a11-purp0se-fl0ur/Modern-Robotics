from Functions.Mia_Functions import *
# Wrenches

# B Frame
x_b = np.array([0, -1, 0])
y_b = np.array([1, 0, 0])
z_b = np.array([0, 0, 1])

Rtb = rotCombine(x_b, y_b, z_b)
print(Rtb)

#Pt from t to b

Pb = [2, 1, 3]

Ttb = constructT(Rtb, ptb)
print(Ttb)

Fb = np.array([-100, 0, 0]) # Force
Ft = np.array([0, 100, 0])

m = 50 # Mass
acc = 10 # Acceleration

Pb = np.array([2, 1, 3])
Pt = Rt @ Pb
print(Pt)

# Wrench in the B Frame
Wb = Wrench(Fb, Pb)
print(Wb)

# Wrench in the T Frame
Wt = Wrench(Ft, Pt)
print(Wt)

# Conceptual questions
# Links and joints
# lecture up to wrenches