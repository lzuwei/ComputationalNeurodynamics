#!/usr/bin/env python

"""
Computational Neurodynamics
Exercise 1

Solves the ODE dy/dt=y (exact solution: y(t)=exp(t)), by numerical
simulation using the Euler method, for two different step sizes.

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.001     # Step size for exact solution
dt_small = 0.1 # Small integration step
dt_large = 0.5 # Large integration step

# Create time points
Tmin = 0       # Time Start
Tmax = 5       # Time End
T = np.arange(Tmin, Tmax+dt, dt)    # create time scale with 0..1 intervals
T_small = np.arange(Tmin, Tmax+dt_small, dt_small)
T_large = np.arange(Tmin, Tmax+dt_large, dt_large)
y = np.zeros(len(T))
y_small = np.zeros(len(T_small))
y_large = np.zeros(len(T_large))

# Exact solution
y = np.exp(T)   # a plot with exact equation using very small intervals

# Approximated solution with small integration Step
# dydx = e^x = y, differentiation rule of e^x
# therefore, euler's formula
# y(t + dt) = y(t) + dt * y(t)
y_small[0] = np.exp(Tmin) # Initial value
for t in xrange(1, len(T_small)):
  y_small[t] = y_small[t-1] + dt_small*y_small[t-1]

# Approximated solution with large integration Step
y_large[0] = np.exp(Tmin) # Initial value
for t in xrange(1, len(T_large)):
  y_large[t] = y_large[t-1] + dt_large*y_large[t-1]

# Plot the results
plt.plot(T      , y      , 'b', label='Exact solution of y = $e^t$')
plt.plot(T_small, y_small, 'g', label='Euler method $\delta$ t = 0.1')
plt.plot(T_large, y_large, 'r', label='Euler method $\delta$ t = 0.5')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()
plt.close()
