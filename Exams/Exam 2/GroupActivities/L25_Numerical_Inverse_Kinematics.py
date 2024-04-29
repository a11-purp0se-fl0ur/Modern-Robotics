from matplotlib import pyplot as plt
from Functions.Mia_Functions import *
dec = 3
np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------------------------------------------------------------------------------
# Example 1

# Define function
def fx(x):
    fx = x**3 - x - 1
    return fx

# ----------------------------------------------------------------------------------------------------------------------
# Plot Function for Initial Guess --------------------------------------------------------------------------------------
# Generate x values
x = np.linspace(-3, 3, 400)  # Generates 400 points between -3 and 3

# Generate y values
y = fx(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = x^3 - x - 1')  # Plot The Function
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
#plt.show()  # Display The Plot
# ----------------------------------------------------------------------------------------------------------------------

# Define derivative of function
def dfx(x):
    dfx = 3 * x**2 - 1
    return dfx

# Initial guess
x0 = 1
epsilon = 1e-5
error = 1e6
iter = 1

# Iterate
print('iter\t xi\t\tfx \t\tfpx \t\terror')
while error > epsilon and iter < 100:
    x1 = x0 - fx(x0)/dfx(x0)
    error = np.abs(x1 - x0)
    print('{:d} \t {:.7f} \t{:.7f} \t{:.7f} \t{:.7f}'.format(iter,x1, fx(x0), dfx(x0), error))
    iter += 1
    x0 = x1

iter -= 1
print('\nSolution:', x1, ' at iteration: ', iter, ' with error: ', error)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Example 2

def fx(x):
    fx = -x**4 + 8*x**2 + 4
    return fx

def dfx(x):
    dfx = -4*x**3 + 16*x
    return dfx

# ----------------------------------------------------------------------------------------------------------------------
# Plot Function for Initial Guess --------------------------------------------------------------------------------------
# Generate x values
x = np.linspace(-3, 3, 400)

# Generate y values
y = fx(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
#plt.show()  # Display The Plot
# ----------------------------------------------------------------------------------------------------------------------

# Initial Guess
x0 = -3
epsilon = 1e-5
error = 1e6
iter = 1

# Iterate
print('iter\t xi\t\tfx \t\tfpx \t\terror')
while error > epsilon and iter < 100:
    x1 = x0 - fx(x0)/dfx(x0)
    error = np.abs(x1 - x0)
    print('{:d} \t {:.7f} \t{:.7f} \t{:.7f} \t{:.7f}'.format(iter,x1, fx(x0), dfx(x0), error))
    iter += 1
    x0 = x1

iter -= 1
print('\nSolution:', x1, ' at iteration: ', iter, ' with error: ', error)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Example 3

def fx(X):
    x,y = X
    fx1 = x ** 2 + y ** 2 - 4
    fx2 = 4 * x ** 2 - y ** 2 - 4
    return np.array([fx1, fx2])

def dfx(X):
    x,y = X
    df1dx = 2 * x
    df1dy = 2 * y
    df2dx = 8 * x
    df2dy = -2 * x
    return np.array([[df1dx, df1dy],[df2dx, df2dy]])

# ----------------------------------------------------------------------------------------------------------------------
# Plot Function for Initial Guess --------------------------------------------------------------------------------------
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)

X, Y = np.meshgrid(x, y)

# Compute the values for each function on the grid
fx1 = X**2 + Y**2 - 4
fx2 = 4*X**2 - Y**2 - 4

plt.figure(figsize=(8, 6))
contour_fx1 = plt.contour(X, Y, fx1, levels=[0], colors='blue')
plt.clabel(contour_fx1, inline=True, fontsize=8, fmt='fx1')
contour_fx2 = plt.contour(X, Y, fx2, levels=[0], colors='red')
plt.clabel(contour_fx2, inline=True, fontsize=8, fmt='fx2')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#plt.show() # Show the plot
# ----------------------------------------------------------------------------------------------------------------------

# Initial guess
X0 = np.array([1, 1])
epsilon = 1e-5
error = 1e6
iter = 1

# Iterate
print('iter\t xi\t\tyi \t\tf(x1) \t\tf(y1) \t\tf(x0) \t\tf(y0) \t\terror')
while error > epsilon and iter < 100:
    X1 = X0 - np.linalg.inv(dfx(X0)) @ fx(X0)
    f1 = fx(X1)
    f2 = fx(X0)
    error = np.linalg.norm(X1 - X0)
    print('{:d} \t {:.7f} \t{:.7f} \t{:.7f} \t{:.7f} \t{:.7f} \t{:.7f} \t{:.7f}'.format(iter,X1[0], X1[1], f1[0], f1[1], f2[0], f2[1], error))
    iter += 1
    X0 = X1

iter -= 1
print('\nSolution:', X1, ' at iteration: ', iter, ' with error: ', error)