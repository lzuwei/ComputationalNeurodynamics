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
dy = np.zeros(len(T))
d2y = np.zeros(len(T))

# Approximated solution with small integration Step
# dydx = e^x = y, differentiation rule of e^x
# therefore, euler's formula
# y(t + dt) = y(t) + dt * y(t)
# set initial conditions and constants:
# y = 1,
# dy/dt = 0
# m = 1, c = 0.1, k = 1
m = 1
c = 0.1
k = 1
y[0] = 1
dy[0] = 0
d2y[0] = (-c/m)*dy[0] - (k/m)*y[0]

for t in xrange(1, len(T_small)):
    dy[t] = dy[t - 1] + d2y[t-1] * dt_small
    d2y[t] = d2y[t-1] + (-1/m * (c * d2y[t-1] + k * dy[t]))

# Plot the results
# plt.plot(T, y, 'b', label='Exact solution of y = $e^t$')
plt.plot(T_small, dy, 'g', label='Euler method $\delta$ t = 0.1')
# plt.plot(T_large, y_large, 'r', label='Euler method $\delta$ t = 0.5')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()
plt.close()
