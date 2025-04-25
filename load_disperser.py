import numpy as np
import hparameters
import os
from scipy import interpolate
from astropy.modeling import models, fitting
import warnings

class MyDisperser():

    def __init__(self, label, As, lambdas, x0, disperser_dir=hparameters.DISPERSER_DIR):

        self.label = label

        # Load transmission
        filename = f"{disperser_dir}/{label}/transmission.txt"
        a = np.loadtxt(filename)
        l, t, _ = a.T
        self.transmission = interpolate.interp1d(l, t, bounds_error=False, fill_value=0.)

        # Load Ratio order
        filename = f"{disperser_dir}/{label}/ratio_order_2over1.txt"
        a = np.loadtxt(filename)
        if a.T.shape[0] == 2 : l, t = a.T
        else : l, t, e = a.T
        self.ratio_order_2over1 = interpolate.interp1d(l, t, bounds_error=False, kind="linear", fill_value="extrapolate")

        filename = f"{disperser_dir}/{label}/ratio_order_3over2.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            if a.T.shape[0] == 2 : l, t = a.T
            else : l, t, e = a.T
            self.ratio_order_3over2 = interpolate.interp1d(l, t, bounds_error=False, kind="linear", fill_value="extrapolate")
            self.ratio_order_3over1 = interpolate.interp1d(l, self.ratio_order_3over2(l)*self.ratio_order_2over1(l), bounds_error=False, kind="linear", fill_value="extrapolate")
        else:
            self.ratio_order_3over2 = None
            self.ratio_order_3over1 = None     
            
        # Load grating
        filename = f"{disperser_dir}/{label}/hologram_grooves_per_mm.txt"
        a = np.loadtxt(filename)
        self.N_x, self.N_y, N_data = a.T
        if hparameters.CCD_REBIN > 1:
            self.N_x /= hparameters.CCD_REBIN
            self.N_y /= hparameters.CCD_REBIN
        self.N_interp = interpolate.CloughTocher2DInterpolator((self.N_x, self.N_y), N_data)
        self.N_fit = self.fit_poly2d(self.N_x, self.N_y, N_data, order=2)

        # make distance_along_disp_axis:
        self.dist_along_disp_axis = list()
        for order, A in enumerate(As):
            if A != 0.0 : self.dist_along_disp_axis.append(self.grating_lambda_to_pixel(lambdas, x0=x0, order=order))
            else : self.dist_along_disp_axis.append(None)

    def grating_lambda_to_pixel(self, lambdas, x0, order=1):

        lambdas = np.copy(lambdas)
        theta0 = (x0[0] - hparameters.CCD_IMSIZE / 2) * hparameters.CCD_PIXEL2ARCSEC * hparameters.CCD_ARCSEC2RADIANS

        theta = np.arcsin(np.clip(order * lambdas * 1e-6 * self.N(x0) + np.sin(theta0),-1, 1))
        deltaX = hparameters.DISTANCE2CCD * (np.tan(theta) - np.tan(theta0)) / hparameters.CCD_PIXEL2MM

        return deltaX

    def N(self, x):

        if x[0] < np.min(self.N_x) or x[0] > np.max(self.N_x) or x[1] < np.min(self.N_y) or x[1] > np.max(self.N_y) : N = float(self.N_fit(*x))
        else : N = int(self.N_interp(*x))

        return N

    def fit_poly2d(self, x, y, z, order):
    
        p_init = models.Polynomial2D(degree=order)
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            p = fit_p(p_init, x, y, z)
            return p
