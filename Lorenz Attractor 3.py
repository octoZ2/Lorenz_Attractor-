import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Formulation of the Lorenz Differentials 

WIDTH, HEIGHT, DPI = 1000, 750, 100

# Pramaters/Initial Conditions 

sigma, beta, rho = 10, 2.667, 28

u0, v0, w0 = 0, 1, 1.05

# Max

tmax, n = 100, 10000

def lorenz(t, X, sigma, beta, rho):
    ''' Lorenz Equations'''
    u, v, w = X

    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v

    return up, vp, wp

# Integration

soln = solve_ivp(lorenz, (0,tmax), (u0, v0, w0), args = (sigma, beta, rho), dense_output = True)

# Interpolate onto t-axis
t = np.linspace(0, tmax, n)
x, y, z = soln.sol(t)

# Plot Lorenz using Matplotlib 
fig = plt.figure(facecolor = 'k', figsize = (WIDTH/DPI, HEIGHT/DPI))
ax = fig.gca(projection = '3d')
ax.set_facecolor('k')
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)

# Mutli coloured segments

s = 10
cmap = plt.cm.autumn
for i in range(0,n-s,s):
    ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(1/n), alpha = 0.4)


ax.set_axis_off ()

plt.savefig('Lorenz.png', dpi = DPI)
plt.show()

