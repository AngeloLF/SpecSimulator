import numpy as np
from numba import njit
import hparameters
from time import time



def simpleLinear(x, a):

    return a



def moffat2d_timbre(gamma, alpha):

    return gamma * 6



def moffat2d(x, y, amplitude, x_c, y_c, gamma, alpha):

    xc = x - x_c
    yc = y - y_c
    rr_gg = (xc * xc + yc * yc) / (gamma * gamma)
    a = (1 + rr_gg) ** -alpha
    norm = (np.pi * gamma * gamma) / (alpha - 1)
    a *= amplitude / norm
    return a



@njit(["float32[:,:](int32[:,:], int32[:,:], float32, float32, float32, float32, float32)",
       "float32[:,:](float32[:,:], float32[:,:], float32, float32, float32, float32, float32)"], fastmath=True, cache=True)
def moffat2d_jit(x, y, amplitude, x_c, y_c, gamma, alpha):

    xc = x - x_c
    yc = y - y_c
    rr_gg = (xc * xc + yc * yc) / (gamma * gamma)
    a = (1 + rr_gg) ** -alpha
    norm = (np.pi * gamma * gamma) / (alpha - 1)
    a *= amplitude / norm
    return a

