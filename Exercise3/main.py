# course: TMR4182
# exercise: 3
# name: Jonas Bjørlo

import numpy as np
import time
import matplotlib.pyplot as plt
from const_avg_acc import const_avg_acc
from runge_kutta4 import runge_kutta4

# Inputs
# Input dynamic system characteristics
m = 6  # mass [kg]
k = 3  # stiffness [N/m]
c = 0.1  # damping [N/s]
u0 = 0.2  # initial position [m]
udot0 = 0.1  # initial velocity [m/s]
loadtype = 2

# Input load parameters
h = 0.1  # time increment [s]
tmax = 500  # time duration [s]
t = np.arange(0, tmax, h)

if (loadtype == 0):  # Harmonic load
    omega = 0.4  # frequency
    amp = 1  # amplitude
    eps1 = np.pi/8  # phase angle
    P = amp*np.cos(omega*t+eps1)  # load vector
elif (loadtype == 1):  # switch load
    omega1 = 2*np.pi/2  # frequency 1
    amp1 = 3  # amplitude 1
    eps1 = 0  # phase 1
    omega2 = 2*np.pi/10  # frequency 2
    amp2 = 1  # amplitude 2
    eps2 = 0  # phase 2
    tswitch = 50  # time for switch
    P = np.zeros(len(t))
    P[t < tswitch] = amp1 * np.cos(omega1*t[t < tswitch]+eps1)
    P[t >= tswitch] = amp2 * np.cos(omega2*t[t >= tswitch]+eps2)
else:  # short duration
    Pmax = 10  # maximum load [N]
    t1 = 0.5  # time duration of load [s]
    omega = 2*np.pi/(2*t1)  # frequency for sine shape
    P = np.zeros(len(t))  # load vector
    indt1 = sum(t >= t1)
    P[0:indt1] = Pmax*np.sin(omega*t[0:indt1])

# Student code goes here (Whatever that means)
caastart = time.time()
u = const_avg_acc(m, k, c, P, t, u0, udot0)
caaend = time.time()

rkstart = time.time()
urk = runge_kutta4(m, k, c, P, t, u0, udot0)
rkend = time.time()

# Plot
plt.figure(1)
plt.subplot(212)
plt.plot(t, u, 'k', label='CAA')
plt.plot(t, urk, 'b--', label='RK')
plt.legend()
plt.grid(True)
plt.xlabel('time, s')

tcaa = caaend-caastart
print(tcaa)

trk = rkend - rkstart
print(trk)
