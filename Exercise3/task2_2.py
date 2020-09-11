import numpy as np
import matplotlib.pyplot as plt
from const_avg_acc import const_avg_acc
from runge_kutta4 import runge_kutta4


def transient_regime(m, k, c, P, t, u0, udot0):
    # Showing effect of the initial conditions
    # and damping on the transient regime
    for i in range(len(c)):
        u_caa = const_avg_acc(m, k, c[i], P, t, u0[i], udot0[i])
        u_rk4 = runge_kutta4(m, k, c[i], P, t, u0[i], udot0[i])

        plt.figure(i)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(t, u_rk4, 'b-', label='RK')
        plt.plot(t, u_caa, '--', color='red', label='CAA')

        plt.legend()
        plt.title(r'$\dot{{u}}_0$ = {:.2f}, $u_0$ = {:.2f}, $c$ = {:.2f}'.format(
            udot0[i], u0[i], c[i]))
        plt.grid(True)
        plt.xlabel('time, s')
        plt.ylabel('u(t)')
    plt.show()
