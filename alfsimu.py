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

var_params_atm = {
	"ATM_AEROSOLS" : [0.0, 0.8],
    "ATM_OZONE" : [225.0, 375.0],
    "ATM_PWV" : [0.0, 15.0],
    "ATM_AIRMASS" : [1.0, 2.5],
}

var_params = {
	'arg.0.0' : [2.0, 8.0],
	'ROTATION_ANGLE' : [-3.0, 3.0],
	"A" : [0.6, 1.4],
}

full_var = {**var_params_atm, **var_params}


if "tsim" in sys.argv:

	delete_ctt("simulator")
	from simulator_true import SpecSimulator

else:

	from simulator import SpecSimulator


if "lsp" not in sys.argv:

	sim = SpecSimulator(psf_function, full_var, input_argv=sys.argv[1:], with_noise=True, output_dir="output_simu", output_fold=f"simulation")
	sim.run()

else:

	for noisy, output_dir in [(False, "output_test"), (True,"output_test_noisy")]:

		for param, rang in full_var.items():

			sim = SpecSimulator(psf_function, {param:rang}, input_argv=sys.argv[1:], with_noise=noisy, output_dir=output_dir, output_fold=f"test_{param}", overwrite=True)
			sim.run()

		sim = SpecSimulator(psf_function, var_params_atm, input_argv=sys.argv[1:], with_noise=noisy, output_dir=output_dir, output_fold=f"test_atm", overwrite=True)
		sim.run()