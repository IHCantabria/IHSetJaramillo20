import numpy as np
from numba import jit

@jit
def jaramillo20(E, dt, a, b, cacr, cero, Yini, vlt):
    """
    Jaramillo et al. 2020 model
    """
    Seq = (E - b) / a
    Y = np.zeros_like(E)
    Y[0] = Yini
    for i in range(0, len(E)-1):
        if Y[i] < Seq[i+1]:
            Y[i+1] = ((Y[i]-Seq[i+1])*np.exp(-1 * a *cacr *(E[i+1] ** 0.5)*dt[i]))+Seq[i+1] + vlt*dt[i]
        else:
            Y[i+1] = ((Y[i]-Seq[i+1])*np.exp(-1 * a *cero *(E[i+1] ** 0.5)*dt[i]))+Seq[i+1] + vlt*dt[i]

    return Y, Seq


def jaramillo20_njit(E, dt, a, b, cacr, cero, Yini, vlt):
    """
    Jaramillo et al. 2020 model
    """
    Seq = (E - b) / a
    Y = np.zeros_like(E)
    Y[0] = Yini
    for i in range(0, len(E)-1):
        if Y[i] < Seq[i+1]:
            Y[i+1] = ((Y[i]-Seq[i+1])*np.exp(-1 * a *cacr *(E[i+1] ** 0.5)*dt[i]))+Seq[i+1] + vlt*dt[i]
        else:
            Y[i+1] = ((Y[i]-Seq[i+1])*np.exp(-1 * a *cero *(E[i+1] ** 0.5)*dt[i]))+Seq[i+1] + vlt*dt[i]

    return Y, Seq