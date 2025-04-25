import numpy as np
from numba import njit
import jax
import jax.numpy as jnp
import hparameters
from time import time



def simpleLinear(x, a):

    return a



def moffat2d_timbre(gamma, alpha):

    return gamma * 3



def moffat2d(x, y, amplitude, x_c, y_c, gamma, alpha):

    xc = x - x_c
    yc = y - y_c
    rr_gg = (xc * xc + yc * yc) / (gamma * gamma)
    a = (1 + rr_gg) ** -alpha
    norm = (np.pi * gamma * gamma) / (alpha - 1)
    a *= amplitude / norm
    return a



@njit(["float32[:,:](int32[:,:], int32[:,:], float32, float32, float32, float32, float32)",
       "float64[:,:](int32[:,:], int32[:,:], float64, float64, float64, float64, float64)",
       "float32[:,:](float32[:,:], float32[:,:], float32, float32, float32, float32, float32)"], fastmath=True, cache=True)
def moffat2d_jit(x, y, amplitude, x_c, y_c, gamma, alpha):

    xc = x - x_c
    yc = y - y_c
    rr_gg = (xc * xc + yc * yc) / (gamma * gamma)
    a = (1 + rr_gg) ** -alpha
    norm = (np.pi * gamma * gamma) / (alpha - 1)
    a *= amplitude / norm
    return a



@jax.jit
def moffat2d_jax(x, y, amplitude, x_c, y_c, gamma, alpha):
    xc = x - x_c
    yc = y - y_c
    rr_gg = (xc * xc + yc * yc) / (gamma * gamma)
    a = (1 + rr_gg) ** -alpha
    norm = (jnp.pi * gamma * gamma) / (alpha - 1)
    a *= amplitude / norm
    return a



if __name__ == '__main__':

    yy, xx = np.mgrid[:40, :40]
    pixels = np.asarray([xx, yy])

    x_c, y_c = hparameters.R0
    ampli = 1.0
    gamma = 3.0
    alpha = 3.0

    Xpix, Ypix = pixels

    nb_loop = 100000

    total = time()

    times = np.zeros(nb_loop)
    for i in range(nb_loop):
        t0 = time()
        moffat2d(Xpix, Ypix, ampli, x_c, y_c, gamma, alpha)
        times[i] = (time() - t0) * 1e6
    print(f"Numpy : {np.mean(times):.1f} ~ {np.std(times):.1f} µs")

    times = np.zeros(nb_loop)
    for i in range(nb_loop):
        t0 = time()
        moffat2d_jit(Xpix, Ypix, ampli, x_c, y_c, gamma, alpha)
        times[i] = (time() - t0) * 1e6
    print(f"Numba : {np.mean(times):.1f} ~ {np.std(times):.1f} µs")

    times = np.zeros(nb_loop)
    for i in range(nb_loop):
        t0 = time()
        moffat2d_jax(Xpix, Ypix, ampli, x_c, y_c, gamma, alpha)
        times[i] = (time() - t0) * 1e6
    print(f"JAX   : {np.mean(times):.1f} ~ {np.std(times):.1f} µs")

    print(time()-total, ' s for total')

