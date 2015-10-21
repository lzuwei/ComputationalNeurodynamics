#!/usr/bin/env python

"""
Computational Neurodynamics
Exercise 1
Question 1

Solves the 2nd Order Differential Equation, by numerical
simulation using the Euler method, for two different step sizes.

For a mass spring system,
d2y/dt2 = -1/m * (c * dy/dt + ky)

To solve using Euler Method, convert 2nd order ODE to a system of
1st order ODEs

Example at:
https://www.youtube.com/watch?v=k2V2UYr6lYw

given the initial conditions,
t = 0,
y = 1,
dy/dt = 0

let d2y/dt2 = z, we have

dy/dt = z, y(0) = 1 ------- (1)
d2y/dt2 = dz/dt = -1/m * (cz + ky), dy/dt = z = 0 ------- (2)

therefore using Euler Method we have 2 ODEs to estimate separately in lock step,
solve y(i), z(i) before solving y(i+1) as y(i+1) requires z(i).

y(i+1) = y(i) + f1(t(i), y(i), z(i)) * (h)
z(i+1) = z(i) + f2(t(i), y(i), z(i)) * (h)

where h is the step size.
so we have :

y(i+1) = y(i) + z(i)*(h)
z(i+1) = z(i) + (-1/m * (cz(i) +ky(i)))*(h)

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.001  # Step size for exact solution
dt_small = 0.1  # Small integration step
dt_large = 0.5  # Large integration step

# Create time points
Tmin = 0  # Time Start
Tmax = 100  # Time End
T = np.arange(Tmin, Tmax + dt, dt)  # create time scale with 0..1 intervals
T_small = np.arange(Tmin, Tmax + dt_small, dt_small)
T_large = np.arange(Tmin, Tmax + dt_large, dt_large)
y = np.zeros(len(T))
y_small = np.zeros(len(T_small))
y_large = np.zeros(len(T_large))

# Exact solution
y = np.exp(T)  # a plot with exact equation using very small intervals

# Approximated solution with small integration Step
# dydx = e^x = y, differentiation rule of e^x
# therefore, euler's formula
# y(t + dt) = y(t) + dt * y(t)
# set initial conditions and constants:
# y = 1,
# dy/dt = 0
# m = 1, c = 0.1, k = 1
y_small[0] = 1
z_curr = 0
m = 1
c = 0.1
k = 1
for t in xrange(1, len(T_small)):
    y_small[t] = y_small[t - 1] + z_curr * dt_small
    z_curr = z_curr + (-1/m * (c * z_curr + k * y_small[t]))

# Approximated solution with large integration Step
y_large[0] = np.exp(Tmin)  # Initial value
for t in xrange(1, len(T_large)):
    y_large[t] = y_large[t - 1] + dt_large * y_large[t - 1]

# Plot the results
# plt.plot(T, y, 'b', label='Exact solution of y = $e^t$')
plt.plot(T_small, y_small, 'g', label='Euler method $\delta$ t = 0.1')
# plt.plot(T_large, y_large, 'r', label='Euler method $\delta$ t = 0.5')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()
plt.close()
