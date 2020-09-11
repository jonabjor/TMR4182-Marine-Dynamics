import numpy as np
import matplotlib.pyplot as plt
from const_avg_acc import const_avg_acc
from runge_kutta4 import runge_kutta4


def response_vs_dlf(m, k, c, w, P_0, t, u0, udot0):
    # Comparing steady state response to the DLF
    w0 = np.sqrt(k/m)  # Natural frequency
    beta = w / w0  # Frequency ratios
    zeta = c / (2*m*w0)  # Damping ratios
    dlf = 1 / ((1-beta)**2 + (2*zeta*beta)**2)**(1/2)  # DLF array values
    phase = np.arctan2(-2*zeta*beta, (1-beta**2))  # Phase angle

    # Plot DLF array against beta
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(beta, dlf, 'o-', label='DLF')
    plt.title("DLF for the chosen data")
    plt.ylabel(r'DLF')
    plt.xlabel(r'$\beta$')
    plt.grid(True)
    plt.show()
    print("Dynamic Load factors: ", dlf)
    print("Phase angles: ", phase)

    for i in range(len(w)):
        P = P_0 * np.cos(w[i]*t)
        u_caa = const_avg_acc(m, k, c, P, t, u0, udot0)
        u_rk4 = runge_kutta4(m, k, c, P, t, u0, udot0)

        plt.figure(i)
        #plt.plot(t, P, color="black", label="P(t)")
        plt.plot(t, u_rk4, 'b-', label='RK')
        plt.plot(t, u_caa, '--', color='red', label='CAA')
        plt.title(r'$\omega$ = {}, DLF = {:.2f}, $\phi$ = {:.2f}'.format(
            w[i], dlf[i], phase[i]))
        plt.legend()
        plt.grid(True)
        plt.xlabel('time, s')
        plt.ylabel('u(t)')

    plt.show()
