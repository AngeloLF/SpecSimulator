import os, sys, shutil, json, pickle
import numpy as np
import matplotlib.pyplot as plt
import coloralf as c
import astropy.units as u
from scipy.interpolate import interp1d
from time import time, ctime
from getCalspec import getCalspec
from tqdm import tqdm

from utils_spec.ctTime import ctTime
from utils_spec.adr import adr_calib
from utils_spec.load_disperser import MyDisperser

import hparameters
if hparameters.SPECTRACTOR_ATMOSPHERE_SIM.lower() == "getobsatmo" : import getObsAtmo



class SpecSimulator():

    """
    ---
    """

    def __init__(self, psf_function, var_params, output_path='./results', output_dir='output_simu', output_fold='simulation', input_argv=list(),
                    with_adr=True, with_atmosphere=True, with_background=True, with_flat=True, with_convertADU=True, with_noise=True,
                    overwrite=True, show_times=True, show_specs=True, target_set="set0", mode4variable="rdm", verbose=2,
                    nb_simu=10, disperser=None):

        """
        verbose :
            * 0 pour aucun print, a par la pbar et le nom du folder
            * 1 pour les print principaux
            * 2 pour tout les prints
        """


        time_init = time()

        # PSF function and output_dir for simulation results.
        self.psf_function = psf_function
        if output_dir not in os.listdir(output_path) : os.mkdir(f"{output_path}/{output_dir}")
        self.output_dir = f"{output_path}/{output_dir}"

        # Parameters define by sys.argv
        self.nb_simu_base = nb_simu
        self.show_times = show_times
        self.show_specs = show_specs
        self.overwrite = overwrite
        self.target_set = target_set
        self.mode4variable = mode4variable
        self.output_fold = output_fold
        self.verbose = verbose
        for argv in input_argv:
            if argv[0] == "x" : self.nb_simu_base = int(argv[1:])
            if argv[1:] == 'times' : self.show_times = True if argv[0] == '+' else False
            if argv[1:] == 'specs' : self.show_specs = True if argv[0] == '+' else False
            if argv[1:] in ['overwrite', 'ow'] : self.overwrite = True if argv[0] == '+' else False
            if argv[:3] == 'set' : self.target_set = argv
            if argv in ["rdm", "lsp"] : self.mode4variable = argv
            if argv[:2] == 'f=' : self.output_fold = argv[2:]
            if argv[:2] == 'v=' : self.verbose = int(argv[2:])
            if argv[:5] == 'disp=' : disperser = argv[5:]

        self.nb_simu = self.nb_simu_base if self.mode4variable == 'rdm' else self.nb_simu_base * len(hparameters.TARGETS_NAME[self.target_set])
        self.len_simu = len(str(self.nb_simu-1))

        # Initialisation
        if self.verbose >= 0: 
            print(f"{c.y}\nInitialisation of SpecSimulator at {c.d}{c.ly}{c.ti}{ctime()}{c.d}{c.d}")

        # Define variables parameters for the simulation
        self.init_var_params(var_params)

        # Define output directory
        num = 0
        self.save_fold = self.output_fold
        if not self.overwrite:
            while self.save_fold in os.listdir(self.output_dir):
                num += 1
                self.save_fold = self.output_fold + "_" + str(num)
        elif self.save_fold in os.listdir(self.output_dir):
            if self.verbose > 0: 
                print(f"{c.y}Overwriting... delete of {self.output_dir}/{self.save_fold}{c.d}")
            shutil.rmtree(f"{self.output_dir}/{self.save_fold}")
        if self.verbose >= 0: 
            print(f"{c.y}Create folder {self.output_dir}/{self.save_fold}{c.d}")
        os.mkdir(f"{self.output_dir}/{self.save_fold}")
        os.mkdir(f"{self.output_dir}/{self.save_fold}/spectrum")
        os.mkdir(f"{self.output_dir}/{self.save_fold}/image")
        os.mkdir(f"{self.output_dir}/{self.save_fold}/divers")
        
        # Order 0 coord.
        self.R0 = hparameters.R0

        # Simulation size
        self.Nx = hparameters.SIM_NX
        self.Ny = hparameters.SIM_NY
        self.yy, self.xx = np.mgrid[:self.Ny, :self.Nx].astype(np.int32)
        self.pixels = np.asarray([self.xx, self.yy])

        # Class ctTime, for detailled execution time

        # Definition of lambdas
        self.N = hparameters.N
        self.lambdas = hparameters.LAMBDAS
        self.lambda_adr_ref = 550

        # Loading Disperser, Amplitude and transmission ratio
        self.disperser_name = disperser if disperser is not None else hparameters.DISPERSER 
        if self.verbose >= 0 : print(f"Loading disperser {c.ti}{self.disperser_name}{c.d}")
        self.As = [0.0, self.A1, self.A2, self.A3]
        self.disperser = MyDisperser(self.disperser_name, self.As, self.lambdas, self.R0)
        self.tr = [None] + self.giveTr()
        self.order2make = {order:[tr, A] for order, (tr, A) in enumerate(zip(self.tr, self.As)) if tr is not None and A != 0.0}

        # Simulation parameters
        self.with_atmosphere = with_atmosphere
        self.with_adr = with_adr
        self.with_background = with_background
        self.with_flat = with_flat
        self.with_convertADU = with_convertADU
        self.with_noise = with_noise
        if self.verbose >= 0 : print(f"With noise : {c.ti}{self.with_noise}{c.d}")

        # Loading telescope transmission
        self.telescope_transmission = self.loading_tel_transmission()

        # Loading target spectrum
        self.target_spectrum = self.loading_target_spectrum(hparameters.TARGETS_NAME[self.target_set])
        
        # Loading flat
        self.flat = self.loading_flat()

        # Loading Atmosphere
        self.load_atm_transmission()

        # Total time for this initialisation
        total_time = time() - time_init
        if self.verbose > 0:
            print(f"{c.y}Initialisation of SpecSimulator : {total_time:.2f} s. {c.d}")


    def run(self):

        times = list()
        pbar = tqdm(total=self.nb_simu)
        for i in range(self.nb_simu):
            t0 = time()
            image, spectrum = self.makeSim(num_simu=i)
            pbar.update(1)
            times.append(time()-t0)
        pbar.close()

        self.json_save(self.historic_params, 'hist_params')
        self.json_save(self.hp_params,  'hparams')

        with open(f"{self.output_dir}/{self.save_fold}/variable_params.pck", "wb") as f:
            pickle.dump(self.variable_params, f)

        if self.show_times:
            nb_train = 50000
            time_per_train = nb_train * (np.sum(times) / self.nb_simu) / 60
            if self.verbose > 0: 
                print(f"{c.lm}Result of ctTime with {self.nb_simu} loop : {np.mean(times)*1e3:.1f} ~ {np.std(times)*1e3:.1f} ms{c.d}") 
            if self.verbose > 1:
                print(f"Time for {nb_train} pict. : {time_per_train:.1f} min with {image.shape[0] * image.shape[1] * 8 / 1024**3 * nb_train:.2f} Go")


        if self.show_specs:

            if self.nb_simu >= 10:

                plt.figure(figsize=(16, 12))
                files = os.listdir(f"{self.output_dir}/{self.save_fold}/image")[:10]

                for i, file in enumerate(files):
                    img = np.load(f"{self.output_dir}/{self.save_fold}/image/{file}")

                    plt.subplot(5, 2, i+1)
                    plt.imshow(np.log(img+1), cmap='gray', origin='lower')
                    plt.title(self.variable_params['TARGET'][i])
                    plt.axis('off')

                plt.savefig(f"{self.output_dir}/{self.save_fold}/divers/images.png")
                plt.close()

                plt.figure(figsize=(24, 12))
                files = os.listdir(f"{self.output_dir}/{self.save_fold}/spectrum")[:10]
                for i, file in enumerate(files):
                    title = [self.variable_params['TARGET'][i]] + [f"{var}={self.variable_params[var][i]:.2f}" for var in self.variable_params.keys() if var!='TARGET' and var[:4]=="ATM_"]
                    spec = np.load(f"{self.output_dir}/{self.save_fold}/spectrum/{file}")
                    plt.plot(self.lambdas, spec, label=', '.join(title))
                plt.legend()
                plt.savefig(f"{self.output_dir}/{self.save_fold}/divers/specs.png")
                plt.close()


    def makeSim(self, num_simu):

        ### set variable params

        for param in self.variable_params.keys():

            # set var params
            if param[:4] != "arg.": 

                self.__setattr__(param, self.variable_params[param][num_simu])

            # set var arg psf
            else:

                num_arg, num_coef = self.var_arg[param]
                self.psf_function['arg'][num_arg][num_coef] = self.variable_params[param][num_simu]


        # set timbre
        arg_timbre = [int(np.round(np.max(f_arg(self.lambdas, *arg)))) for f_arg, arg in zip(self.psf_function['f_arg'], self.psf_function['arg'])]
        timbre_size = self.psf_function['timbre'](*arg_timbre)

        # SIMULATE
        spectrogram_data = np.zeros((self.Ny, self.Nx), dtype="float32")
        
        spectrum = self.simulate_spectrum() * self.A

        for order, (tr, A) in self.order2make.items():
            
            # Dispersion law
            adr_x, adr_y = self.loading_adr()
            Amp = A * tr(self.lambdas) * spectrum
            X_c = self.disperser.dist_along_disp_axis[order] * np.cos(self.ROTATION_ANGLE * np.pi / 180) + adr_x + self.R0[0]
            Y_c = self.disperser.dist_along_disp_axis[order] * np.sin(self.ROTATION_ANGLE * np.pi / 180) + adr_y + self.R0[1]

            # Building PSF
            spectrogram_data += self.build_psf_cube(X_c, Y_c, Amp, timbre_size)


        # IMAGE RECOMBINAISON
        data_image = spectrogram_data + self.psf_function['f'](self.xx, self.yy, self.psf_function['order0']['amplitude']*self.A0*self.A, *self.R0, *self.psf_function['order0']['arg']).astype(np.float32)

        if self.with_background:
            data_image += self.BACKGROUND_LEVEL

        if self.with_flat:
            data_image *= self.flat

        if self.with_convertADU:
            data_image *= self.EXPOSURE

        if self.with_noise:
            data_image = self.add_poisson_and_read_out_noise(data_image)

        if hparameters.OBS_NAME == "AUXTEL" : data_image = data_image.T[::-1, ::-1]
        np.save(f"{self.output_dir}/{self.save_fold}/image/image_{num_simu:0{self.len_simu}}.npy", data_image)
        np.save(f"{self.output_dir}/{self.save_fold}/spectrum/spectrum_{num_simu:0{self.len_simu}}.npy", spectrum.astype(np.float32))

        return data_image, spectrum



    def simulate_spectrum(self):

        if self.with_atmosphere : self.atm = self.give_atm_transmission()

        spectrum = self.targets_spectrum[self.TARGET](self.lambdas)
        spectrum *= self.disperser.transmission(self.lambdas)
        spectrum *= self.telescope_transmission(self.lambdas)
        if self.with_atmosphere : spectrum *= self.atm(self.lambdas)
        spectrum *= hparameters.FLAM_TO_ADURATE * self.lambdas * np.gradient(self.lambdas)

        return spectrum



    def build_psf_cube(self, X_c, Y_c, amplitude, timbre_size, dtype="float32"):


        argmin = max(0,  int(np.argmin(np.abs(X_c))))
        argmax = min(self.Nx, np.argmin(np.abs(X_c-self.Nx)) + 1)
        psf_cube = np.zeros((hparameters.SIM_NY, hparameters.SIM_NX), dtype=dtype)    
        timbreX = np.zeros((int(timbre_size*2), int(timbre_size*2)), dtype=dtype)
        timbreY = np.zeros((int(timbre_size*2), int(timbre_size*2)), dtype=dtype)

        for x in range(argmin, argmax):

            xmin = max(0, int(X_c[x]                  - timbre_size))
            xmax = min(hparameters.SIM_NX, int(X_c[x] + timbre_size))
            ymin = max(0, int(Y_c[x]                  - timbre_size))
            ymax = min(hparameters.SIM_NY, int(Y_c[x] + timbre_size))

            Xpix, Ypix = self.pixels[:, ymin:ymax, xmin:xmax]
            timbreX[:ymax-ymin, :xmax-xmin] = Xpix
            timbreY[:ymax-ymin, :xmax-xmin] = Ypix
            argf = [f_arg(self.lambdas[x], *arg) for f_arg, arg in zip(self.psf_function['f_arg'], self.psf_function['arg'])]

            psf_cube[ymin:ymax, xmin:xmax] += self.psf_function['f'](timbreX, timbreY, amplitude[x], X_c[x], Y_c[x], *argf)[:ymax-ymin, :xmax-xmin]

        return psf_cube



    ##### 
    #####    Fonction d'initialisation
    #####



    def init_var_params(self, var_params):

        # Pour ce souvenir des paramètres utlisé
        self.historic_params = {'nb_simu':self.nb_simu, 'target_set':hparameters.TARGETS_NAME[self.target_set], 'mode4variable':self.mode4variable}

        # On calcule les vecteurs pour les paramètres variables
        if self.mode4variable == 'lsp':
            # Mode linspace
            nb_target = len(hparameters.TARGETS_NAME[self.target_set])
            self.variable_params = {'TARGET' : np.concatenate(np.array([[targ]*self.nb_simu_base for targ in hparameters.TARGETS_NAME[self.target_set]])).astype(str)}
            func4variable = np.linspace
        elif self.mode4variable == 'rdm':
            # Mode random uniform
            self.variable_params = {'TARGET' : np.random.choice(hparameters.TARGETS_NAME[self.target_set], self.nb_simu)}
            func4variable = np.random.uniform
        else:
            # Mode inexistant
            func4variable = None
            print(f"{c.r}WARNING : mode {self.mode4variable} not exist. Should be `rdm` or `lsp`.{c.d}")

        # On parcoure toute les parametres de hparameters qui peuvent etre variable
        for param, value in hparameters.VARIABLE_PARAMS.items():

            # Les parametres mis en variable dans le dict d'entrée
            if param in var_params and isinstance(var_params[param], (list)):

                if self.verbose > 1: 
                    print(f"Set var param {c.lm}{param}{c.d} to range {c.lm}{var_params[param]}{c.d}")

                self.historic_params[param] = var_params[param]
                if   self.mode4variable == 'lsp' : self.variable_params[param] = np.concatenate([func4variable(*var_params[param], self.nb_simu_base) for _ in range(nb_target)])
                elif self.mode4variable == 'rdm' : self.variable_params[param] = func4variable(*var_params[param], self.nb_simu)

            # Les variables renseigné dans le dict d'entrée mais non variables
            elif param in var_params and isinstance(var_params[param], (int, float)):

                if self.verbose > 1: 
                    print(f"Set fix param {c.m}{param}{c.d} to {c.m}{var_params[param]}{c.d} (from var_params)")
                self.historic_params[param] = var_params[param]
                self.__setattr__(param, var_params[param])

            # Les varaibles non renseigné, on met donc la valeur de défault situé dans hparameters
            else:

                self.__setattr__(param, value)
                self.historic_params[param] = value

        # On s'occupe ici des variation des paramètres de moffat
        var_args = [var for var in var_params if var[:4] == 'arg.']
        self.var_arg = dict()

        for param in var_args:

            if self.verbose > 1: 
                print(f"Set var argu. {c.lm}{param}{c.d} to range {c.lm}{var_params[param]}{c.d}")

            num_arg, num_coef = param.split('.')[1:]
            self.var_arg[param] = [int(num_arg), int(num_coef)]
            self.historic_params[param] = var_params[param]
            if   self.mode4variable == 'lsp' : self.variable_params[param] = np.concatenate([func4variable(*var_params[param], self.nb_simu) for _ in range(nb_target)])
            elif self.mode4variable == 'rdm' : self.variable_params[param] = func4variable(*var_params[param], self.nb_simu)            

        # Enfin, on garde en mémoire tout les autres variables présente dans hparameters
        self.hp_params = dict()

        for hp in dir(hparameters):

            if '__' not in hp and isinstance(hparameters.__getattribute__(hp), (int, str, float)):
                self.hp_params[hp] = hparameters.__getattribute__(hp)



    def loading_tel_transmission(self, throughput_dir=hparameters.THROUGHPUT_DIR):
        """
        Méthode pour load la transmission du telescope 
        """

        filename = f"{throughput_dir}/{hparameters.THROUGHPUT}"
        data = np.loadtxt(filename).T
        lambdas = data[0]
        sorted_indices = lambdas.argsort()
        lambdas = lambdas[sorted_indices]
        y = data[1][sorted_indices]
        indexes = np.logical_and(lambdas > np.min(self.lambdas), lambdas < np.max(self.lambdas))

        to = interp1d(lambdas[indexes], y[indexes], kind='linear', bounds_error=False, fill_value=0.)
        TF = lambda x: 1
        transmission = lambda x: to(x) * TF(x)

        return transmission



    def loading_target_spectrum(self, targets):
        """
        Méthode pour load les spectre des target dans la liste targets
        """

        self.targets_spectrum = dict()

        if self.verbose > 1 : 
            sys.stdout.write(f"Loading targets spectrum : ")
            sys.stdout.flush()

        for target in targets:

            if getCalspec.is_calspec(target):

                calspec = getCalspec.Calspec(target)
                spec_dict = calspec.get_spectrum_numpy()
                spec_dict["WAVELENGTH"] = spec_dict["WAVELENGTH"].to(u.nm)

                for key in ["FLUX", "STATERROR", "SYSERROR"]:
                    spec_dict[key] = spec_dict[key].to(u.erg / u.second / u.cm**2 / u.nm)

                wavelengths = spec_dict["WAVELENGTH"].value
                spectra = spec_dict["FLUX"].value

            else:

                print(f"{c.r}WARNING : label {target} for loading spectrum is not avaible ...{c.d}")
                wavelengths, spectra = None, None

            sed = interp1d(wavelengths, spectra, kind='linear', bounds_error=False, fill_value=0.)

            self.targets_spectrum[target] = sed
            if self.verbose > 1 :
                sys.stdout.write(f"{c.g}{target}{c.d}, ")
                sys.stdout.flush()

        if self.verbose > 1:
            print(f" ... ok")


    def giveTr(self, order=1):

        # load the disperser relative transmissions
        tr_ratio = interp1d(self.lambdas, np.ones_like(self.lambdas), bounds_error=False, fill_value=1.)
        if abs(order) == 1:
            tr_ratio_next_order = self.disperser.ratio_order_2over1
            tr_ratio_next_next_order = self.disperser.ratio_order_3over1
        elif abs(order) == 2:
            tr_ratio_next_order = self.disperser.ratio_order_3over2
            tr_ratio_next_next_order = None
        elif abs(order) == 3:
            tr_ratio_next_order = None
            tr_ratio_next_next_order = None
        else:
            raise ValueError(f"{abs(self.order)=}: must be 1, 2 or 3. "
                             f"Higher diffraction orders not implemented yet in full forward model.")
        
        return [tr_ratio, tr_ratio_next_order, tr_ratio_next_next_order]



    def loading_flat(self, list_gains=[[1]], randomness_level=1e-2, dtype="float32"):
        """
        Méthode pour faire le flat
        """

        gains = np.atleast_2d(list_gains).astype(float)
        if np.mean(gains) != 1. : true_gains /= np.mean(gains)

        flat = np.ones((self.Ny, self.Nx), dtype=dtype)

        hflats = np.array_split(flat, gains.shape[0])

        for h in range(gains.shape[0]):
            vflats = np.array_split(hflats[h].T, gains.shape[1])
            for v in range(gains.shape[1]):
                vflats[v] *= gains[h, v]
            hflats[h] = np.concatenate(vflats).T

        flat = np.concatenate(hflats)
        if randomness_level != 0:
            flat += np.random.uniform(-randomness_level, randomness_level, size=flat.shape)

        return flat



    #####
    #####    Function *utils*
    #####



    def loading_adr(self, dispersion_axis_angle=0):
        """
        Méthode pour load l'ADR
        """

        if self.with_adr:

            ADR_PARAMS = [self.ADR_DEC, self.ADR_HOUR_ANGLE, self.ATM_TEMPERATURE, hparameters.OBS_PRESSURE, self.ATM_HUMIDITY, self.ATM_AIRMASS]
            adr_ra, adr_dec = adr_calib(self.lambdas, ADR_PARAMS, hparameters.OBS_LATITUDE, lambda_ref=self.lambda_adr_ref)

            # flip_and_rotate_radec_vector_to_xy_vector of 
            flip = np.array([[hparameters.OBS_CAMERA_RA_FLIP_SIGN, 0], [0, hparameters.OBS_CAMERA_DEC_FLIP_SIGN]], dtype=float)
            a = - hparameters.OBS_CAMERA_ROTATION * np.pi / 180 # minus sign as rotation matrix is apply on the right on the adr vector
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=float)
            transformation = flip @ rotation
            adr_x, adr_y = (np.asarray([adr_ra, adr_dec]).T @ transformation).T

            # flip_and_rotate_adr_to_image_xy_coordinates
            if not np.isclose(dispersion_axis_angle, 0, atol=0.001):
                # minus sign as rotation matrix is apply on the right on the adr vector
                a = - dispersion_axis_angle * np.pi / 180
                rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=float)
                adr_x, adr_y = (np.asarray([adr_x, adr_y]).T @ rotation).T

            # self.adr_x, self.adr_y = flip_and_rotate_adr_to_image_xy_coordinates(self.adr_ra, self.adr_dec, dispersion_axis_angle=0)
        else:
            adr_x = np.zeros_like(self.disperser.dist_along_disp_axis)
            adr_y = np.zeros_like(self.disperser.dist_along_disp_axis)

        return adr_x, adr_y

    def npy_save(self, dico, savefile):

        np.save(f"{self.output_dir}/{self.save_fold}/{savefile}.npy", dico)



    def npy_load(self, dico, savefile):

        return np.load(f"{self.output_dir}/{self.save_fold}/{savefile}.npy")



    def json_save(self, dico, savefile):

        with open(f"{self.output_dir}/{self.save_fold}/{savefile}.json", 'w') as f:
            json.dump(dico, f, indent=4)



    def json_load(self, savefile):

        with open(f"{self.output_dir}/{self.save_fold}/{savefile}.json", 'r') as f:
            dico = json.load(f)

        return dico



    def add_poisson_and_read_out_noise(self, data):  # pragma: no cover

        d = np.copy(data).astype(np.float32)
        # convert to electron counts
        d *= hparameters.CCD_GAIN
        # Poisson noise
        noisy = np.random.poisson(d).astype(np.float32)
        # Add read-out noise is available
        if self.CCD_READ_OUT_NOISE is not None:
            noisy += np.random.normal(scale=self.CCD_READ_OUT_NOISE*np.ones_like(noisy)).astype(np.float32)
        # reconvert to ADU
        data = noisy / hparameters.CCD_GAIN
        # removes zeros
        min_noz = np.min(data[data > 0])
        data[data <= 0] = min_noz
        return data



    def load_atm_transmission(self):

        if hparameters.SPECTRACTOR_ATMOSPHERE_SIM.lower() == "getobsatmo":

            if not getObsAtmo.is_obssite(hparameters.OBS_NAME):
                raise ValueError(f"getObsAtmo does not have observatory site {hparameters.OBS_NAME}.")

            self.emulator = getObsAtmo.ObsAtmo(obs_str=hparameters.OBS_NAME, pressure=hparameters.OBS_PRESSURE)
            self.emulator.lambda0 = 500.

            return self.give_atm_transmission

        else:

            raise ValueError(f"Unknown value for {hparameters.SPECTRACTOR_ATMOSPHERE_SIM=}.")


        return transmission

    def give_atm_transmission(self):

        atm = self.emulator.GetAllTransparencies(self.lambdas, am=self.ATM_AIRMASS, pwv=self.ATM_PWV, oz=self.ATM_OZONE, tau=self.ATM_AEROSOLS, beta=self.ATM_ANGSTROM_EXPONENT, flagAerosols=True)
        return interp1d(self.lambdas, atm, kind='linear', bounds_error=False, fill_value=(0, 0))