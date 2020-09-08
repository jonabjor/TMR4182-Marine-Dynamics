import numpy as np


def const_avg_acc(m, k, c, P, t, u0, udot0):
    # Compute ü_0 from initial conditions using the eqn. of motion
    uacc0 = (P[0] - c * udot0 - k * u0) / m

    # Initialize arrays and add initial conditions
    u = np.zeros(len(t))
    u[0] = u0

    udot = np.zeros(len(t))
    udot[0] = udot0

    uacc = np.zeros(len(t))
    uacc[0] = uacc0

    # Time vector has constant spacing, therefore we can compute h outside iteration
    h = t[1] - t[0]

    # Computing the responses using constant average acc method
    # The left term from compendium eqn (3.80) is constant and therefore computed outside iteration
    leftterm = ((4/h**2)*m + (2/h)*c + k)

    for i in range(len(t)-1):
        # Finding the reponse u_i+1
        u[i+1] = (P[i+1] + m * uacc[i] + ((4 / h) * m + c) * udot[i] +
                  ((4/h**2) * m + (2/h)*c) * u[i]) / leftterm
        # Computing ü_i+1
        uacc[i+1] = (4 / h**2)*(u[i+1] - u[i] - udot[i]*h) - uacc[i]
        # Computing udot_i+1
        udot[i+1] = udot[i] + (1 / 2)*(uacc[i] + uacc[i+1])*h
    return u
