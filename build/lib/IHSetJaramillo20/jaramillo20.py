import numpy as np
from numba import jit

@jit
def jaramillo20(E, dt, a, b, cacr, cero, Yini, vlt):
    """
    Jaramillo et al. 2020 model
    """
    Seq = (E -b)/a
    Y = np.zeros_like(E)
    Y[0] = Yini
    for i in range(1, len(E)):
        if Y[i-1] < Seq[i]:
            Y[i] = ((Y[i-1]-Seq[i])*np.exp(-1 * a *cacr *(E[i] ^ 0.5)*dt))+Seq[i] + vlt*dt
        else:
            Y[i] = ((Y[i-1]-Seq[i])*np.exp(-1 * a *cero *(E[i] ^ 0.5)*dt))+Seq[i] + vlt*dt

    return Y