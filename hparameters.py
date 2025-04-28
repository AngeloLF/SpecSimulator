import numpy as np


def __getattr__(name):
    """

    """
    if name in locals():
        return locals()[name]
    else:
        raise ValueError(f"hparameters dont have {name=}.")

VARIABLE_PARAMS = {

    # Facteur Amplitude of order:
    "A0" : 1.0, # order 0
    "A1" : 1.0, # order 1
    "A2" : 1.0, # order 2
    "A3" : 0.0, # order 3
    "A" : 1.0, # all order
    
    # Angle of dispersion
    "ROTATION_ANGLE" : 0.0,

    # For atmoshere
    "ATM_AEROSOLS" : 0.05, # 0.00 et 1.00
    "ATM_OZONE" : 300.0, # 250 et 350
    "ATM_PWV" : 5.0, # 0.0 et 15.0
    "ATM_AIRMASS" : 1.2, # 1.0 et 2.5
    "ATM_ANGSTROM_EXPONENT" : 1.2,
    "ATM_TEMPERATURE" : 8.6,
    "ATM_HUMIDITY" : 25,

    # For ADR
    "ADR_DEC" : -18.6,
    "ADR_HOUR_ANGLE" : 28.2,

    # Divers
    "EXPOSURE" : 120.0,
    "BACKGROUND_LEVEL" : 0.5,
    "CCD_READ_OUT_NOISE" : 12.0,
}     

# Targets for spectrums simulations
TARGETS_NAME = {
    # 20 Calspec to use
    "set0" : ['HD009051', 'HD031128', 'HD101452', 'HD106252', 'HD111980', 
            'HD115169', 'HD142331', 'HD14943',  'HD158485', 'HD159222', 
            'HD160617', 'HD163466', 'HD165459', 'HD167060', 'HD185975', 
            'HD200654', 'HD205905', 'HD209458',  'HD37962',  'HD38949'],
            
    # 11 Calspec en +
    "set1" : ['HD074000', 'HD116405', 'HD128998', 'HD172728', 'HD180609', 
              'HD2811',   'HD37725',  'HD55677',  'HD60753',  'HD93521'], #HD200775

    # THE Calspec
    "set2" : ['HD111980'],
}

# SIMULATION PARAMS
SIM_NX = 1024
SIM_NY = 128
R0 = [64, SIM_NY/2]

# LAMBDA PARAMS
LAMBDA_MIN = 300
LAMBDA_MAX = 1100
LAMBDA_STEP = 1
LAMBDAS = np.arange(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP)
N = len(LAMBDAS)

# Divers
FLAM_TO_ADURATE = 1_067_400_516_204.6393
SPECTRACTOR_ATMOSPHERE_SIM = "getobsatmo"

# dec, hour_angle, temperature, pressure, humidity, airmass
# ADR_PARAMS = [-18.6, 28.2, TEMPERATURE, PRESSURE, HUMIDITY, AIRMASS] 

# Disperser characteristics
DISPERSER_DIR = "./datafile/dispersers"
DISPERSER = "HoloAmAg"

# CCD characteristics
CCD_IMSIZE = 2048  # size of the image in pixel
CCD_PIXEL2MM = 24e-3  # pixel size in mm
CCD_PIXEL2ARCSEC = 0.401  # pixel size in arcsec
CCD_ARCSEC2RADIANS = 1 / 180 * np.pi / 3600 # ???
CCD_MAXADU = 60000  # approximate maximum ADU output of the CCD
CCD_GAIN = 3.  # electronic gain : elec/ADU
CCD_REBIN = 1  # rebinning of the image in pixel

# Instrument characteristics
THROUGHPUT_DIR = "./datafile/throughput"
THROUGHPUT = "CTIOThroughput/ctio_throughput_300517_v1.txt"
OBS_NAME = 'CTIO'
OBS_ALTITUDE = 2.200  # CTIO altitude in k meters from astropy package (Cerro Pachon)
OBS_LATITUDE = '-30 10 07.90'  # CTIO latitude
OBS_SURFACE = 6361  # Effective surface of the telescope in cm**2 accounting for obscuration
OBS_EPOCH = "J2000.0"
OBS_OBJECT_TYPE = 'STAR'  # To choose between STAR, HG-AR, MONOCHROMATOR
OBS_FULL_INSTRUMENT_TRANSMISSON = 'ctio_throughput_300517_v1.txt' # Full instrument transmission file
OBS_TRANSMISSION_SYSTEMATICS = 0.005
OBS_CAMERA_ROTATION = 0  # Camera (x,y) rotation angle with respect to (north-up, east-left) system in degrees
OBS_CAMERA_DEC_FLIP_SIGN = 1  # Camera (x,y) flip signs with respect to (north-up, east-left) system
OBS_CAMERA_RA_FLIP_SIGN = 1  # Camera (x,y) flip signs with respect to (north-up, east-left) system
OBS_PRESSURE = 784.0

# Spectrograph characteristics
DISTANCE2CCD = 55.45  # distance between hologram and CCD in mm
DISTANCE2CCD_ERR = 0.19  # uncertainty on distance between hologram and CCD in mm
GRATING_ORDER_2OVER1 = 0.1  # default value for order 2 over order 1 transmission ratio