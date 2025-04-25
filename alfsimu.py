import sys
import psf_func as pf
from ctTime import delete_ctt

psf_function = {
	# For l lambdas in nm :
	# f : def func of (XX, YY, amplitude, x, y, f_argv[0](l, *argv[0]), ..., f_argv[n](l, *argv[n]))
	'f' : pf.moffat2d_jit,

	# function for argument
	'f_arg' : [pf.simpleLinear, pf.simpleLinear],
	
	# argument for argument function
	'arg' : [[3.0], [3.0]],

	# argument order 0
	'order0' : {'amplitude':22900.0, 'arg':[3.0, 2.0]},

	# timbre size function
	'timbre' : pf.moffat2d_timbre,
}

var_params = {
	'arg.0.0' : [2.0, 6.0],
	'ROTATION_ANGLE' : [-2.0, 2.0],
	"ATM_AEROSOLS" : [0.1, 0.7],
    "ATM_OZONE" : [250.0, 350.0],
    "ATM_PWV" : [0.0, 15.0],
    "ATM_AIRMASS" : [1.0, 2.5],
    "A" : [0.8, 1.2],
}


if "tsim" in sys.argv:

	delete_ctt("simulator")
	from simulator_true import SpecSimulator

else:

	from simulator import SpecSimulator


sim = SpecSimulator(psf_function, var_params, input_argv=sys.argv[1:], with_noise=True)
sim.run()