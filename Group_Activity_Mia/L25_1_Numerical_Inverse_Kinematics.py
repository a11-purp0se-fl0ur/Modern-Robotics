from matplotlib import pyplot as plt
from Functions.Mia_Functions import *
dec = 3
np.set_printoptions(precision=3, suppress=True)

# Example 1
def fx(x):
    fx = x**3 - x - 1 # function
    return fx

def fpx(x):
    fpx = 3*x**2 - 1  # derivative
    return fpx

x0 = 1 # initial guess
epsilon = 1e-5
error = 1e6
iter = 1
print('iter\t xi\t\tfx \t\tfpx \t\terror')
while error > epsilon and iter < 100:
    x1 = x0 - fx(x0)/fpx(x0)
    error = np.abs(x1 - x0)
    print('{:d} \t {:.7f} \t{:.7f} \t{:.7f} \t{:.7f}'.format(iter,x1, fx(x0), fpx(x0), error))
    iter += 1
    x0 = x1

iter -= 1
print('\nSolution:', x1, ' at iteration: ', iter, ' with error: ', error)

# Plot the function to verify the root
x = np.linspace(0,2)
y = fx(x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()