import sys
import utils_spec.psf_func as pf
from utils_spec.ctTime import delete_ctt



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
	"train" : {
		"ATM_AEROSOLS" : [0.1, 0.8],
		"ATM_OZONE" : [250.0, 350.0],
		"ATM_PWV" : [0.0, 10.0],
		"ATM_AIRMASS" : [1.2, 2.0]
	},
	"test" : {
		"ATM_AEROSOLS" : [0.0, 0.9],
		"ATM_OZONE" : [220.0, 380.0],
		"ATM_PWV" : [0.0, 15.0],
		"ATM_AIRMASS" : [1.0, 2.5]
	}
}

var_params = {
	"train" : {
		'arg.0.0' : [3.0, 8.0],
		'ROTATION_ANGLE' : [-2.0, 2.0],
		"A" : [0.8, 1.2],
	},
	"test" : {
		'arg.0.0' : [2.0, 10.0],
		'ROTATION_ANGLE' : [-3.0, 3.0],
		"A" : [0.6, 1.4],
	}
}

full_var_train = {**var_params_atm["train"], **var_params["train"]}
full_var_test = {**var_params_atm["test"], **var_params["test"]}


if "tsim" in sys.argv or "new_tsim" in sys.argv:

	if "new_tsim" in sys.argv :
		delete_ctt("simulator")
		
	from simulator_true import SpecSimulator

else:

	from simulator import SpecSimulator


noisy = False if "noisyless" in sys.argv else True 

if "test" not in sys.argv:

	sim = SpecSimulator(psf_function, full_var_train, input_argv=sys.argv[1:], with_noise=noisy, output_dir="output_simu", output_fold=f"simulation")
	sim.run()

else:

	if "lsp" not in sys.argv:

		sim = SpecSimulator(psf_function, full_var_test, input_argv=sys.argv[1:], with_noise=noisy, output_dir="output_simu", output_fold=f"simulation")
		sim.run()

	else:

		for param, rang in full_var_test.items():

			sim = SpecSimulator(psf_function, {param:rang}, input_argv=sys.argv[1:], with_noise=noisy, output_dir="output_test", output_fold=f"test_{param}", overwrite=True)
			sim.run()

		sim = SpecSimulator(psf_function, var_params_atm["test"], input_argv=sys.argv[1:], with_noise=noisy, output_dir="output_test", output_fold=f"test_atm", overwrite=True)
		sim.run()