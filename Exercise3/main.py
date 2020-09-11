# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
from const_avg_acc import const_avg_acc
from runge_kutta4 import runge_kutta4

from task2_1 import response_vs_dlf  # Task 2.1
from task2_2 import transient_regime  # Task 2.2

plt.close('all')

# Input dynamic system characteristics
m = 3  # mass, kg
k = 10  # stiffness, N/m
c = 0.5  # damping, N/s

# Initial conditions
u0 = 0
udot0 = 0.0

# Input load parameters
h = 0.05  # time increment, s
tmax = 60  # time duration, s
t = np.arange(0, tmax, h)

loadtype = 1

t = np.arange(0, tmax, h)

P = np.zeros(shape=(np.size(t)))


if loadtype == 1:  # harmonic load
    omega = np.pi/8      # frequency
    amp = 1               # amplitude
    eps1 = np.pi/8       # phase angle
    P = amp*np.cos(omega*t+eps1)

elif loadtype == 2:  # switch load
    omega1 = 2*np.pi/1    # frequency 1
    amp1 = 3              # amplitude 1
    eps1 = 0              # phase 1
    omega2 = 2*np.pi/10   # frequency 2
    amp2 = 1              # amplitude 2
    eps2 = 0              # phase 2
    tswitch = 50          # time for switch
    P[t <= tswitch] = amp1*np.cos(omega1*t[t <= tswitch]+eps1)
    P[t > tswitch] = amp2*np.cos(omega2*t[t > tswitch]+eps2)

else:  # Short duration, impulse
    Pmax = 10             # maximum load, N
    t1 = 0.5              # time duration of load, s
    omega = 2*np.pi/(2*t1)
    t0 = 0
    indt0 = sum(t <= t0)
    indt1 = sum(t <= t1+t0)
    P[indt0:indt1] = Pmax*np.sin(omega*t[indt0:indt1])

# Calling functions and calculating computing time
caastart = time.time()
u = const_avg_acc(m, k, c, P, t, u0, udot0)
caaend = time.time()

rkstart = time.time()
urk = runge_kutta4(m, k, c, P, t, u0, udot0)
rkend = time.time()

# Simple harmonic load
'''
# Calls for task 2.1
P_0 = 1  # Load amplitude
w = np.array([1.3, 1.5, 1.7, 2.0, 2.3])  # Load frequency
response_vs_dlf(m, k, c, w, P_0, t, u0, udot0)  # Comparing SS response and DLF
'''
# Calls for task 2.2
u0 = [0.0, 0.0, 0.0, 0.0, 0.9, 1.7]
udot0 = [0.0, 0.0, 1.0, 10.0, 0.0, 0.0]
c = [0.5, 7, 1, 1, 1, 1]

# Load settings
omega = np.pi/8      # frequency
amp = 1               # amplitude
P = amp*np.cos(omega*t)  # load P

transient_regime(m, k, c, P, t, u0, udot0)  # Effect of u0, udot0, c
