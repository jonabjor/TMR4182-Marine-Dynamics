# Filename: runge_kutta4.py
import numpy as np


def runge_kutta4(m, k, c, P, t, u0, udot0):
    # Response vector (displacement and velocity)
    u = np.zeros(len(t))
    v = np.zeros(len(t))
    u[0] = u0
    v[0] = udot0

    # dx/dt derivative functions to be called from iteration
    def f1(t, u, v): return v
    def f2(t, u, v, P): return (1/m)*(P-c*v-k*u)

    # Time step
    h = t[1]-t[0]

    # Runge-Kutta 4th order method
    for i in range(len(t)-1):
        # Computing K1, K2, K3 and K4 with f1 and f2
        K0 = np.array([f1(t[i], u[i], v[i]), f2(t[i], u[i], v[i], P[i])])

        K1 = np.array([f1(t[i] + h/2, u[i] + h/2 * K0[0], v[i] + h/2 * K0[1]),
                       f2(t[i] + h/2, u[i] + h/2 * K0[0], v[i] + h/2 * K0[1], (P[i] + P[i+1])/2)])

        K2 = np.array([f1(t[i] + h/2, u[i] + h/2 * K1[0], v[i] + h/2 * K1[1]),
                       f2(t[i] + h/2, u[i] + h/2 * K1[0], v[i] + h/2 * K1[1], (P[i] + P[i+1])/2)])

        K3 = np.array([f1(t[i] + h, u[i] + h * K2[0], v[i] + h * K2[1]),
                       f2(t[i] + h, u[i] + h * K2[0], v[i] + h * K2[1], P[i+1])])

        # Computing u and v from the weighted average given in the compendium
        u[i+1] = u[i] + h/6*(K0[0] + 2*K1[0] + 2*K2[0] + K3[0])
        v[i+1] = v[i] + h/6*(K0[1] + 2*K1[1] + 2*K2[1] + K3[1])

    return u
