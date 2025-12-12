import sys
import coloralf as c
from hparams import Hparams
from specsimulator import SpecSimulator


var_params_atm = {
	"train" : {
		"ATM_AEROSOLS" : [0.1, 0.8],
		"ATM_OZONE" : [250.0, 350.0],
		"ATM_PWV" : [2.0, 10.0],
		"ATM_AIRMASS" : [1.2, 2.0]
	},
	"test" : {
		"ATM_AEROSOLS" : [0.0, 0.9],
		"ATM_OZONE" : [220.0, 380.0],
		"ATM_PWV" : [0.0, 12.0],
		"ATM_AIRMASS" : [1.0, 2.2]
	}
}

var_params = {
	"train" : {
		'arg.0.0' : [3.0, 8.0],
		'ROTATION_ANGLE' : [-0.1, 0.1],
		"A" : [0.8, 1.2],
	},
	"test" : {
		'arg.0.0' : [2.0, 10.0],
		'ROTATION_ANGLE' : [-3.0, 3.0],
		"A" : [0.6, 1.4],
	},
	"perf" : {
		'arg.0.0' : [2.0, 3.0],
		'A' : [1.5, 2.0]
	}
}

full_var_train = {**var_params_atm["train"], **var_params["train"]}
full_var_test = {**var_params_atm["test"], **var_params["test"]}
noisy = False if "noisyless" in sys.argv else True 

if "perfect" in sys.argv:

	hp = Hparams(var_params=var_params["perf"], with_noise=noisy)
	print(f"{c.y}INFO : perfect var params selected ...{c.d}")
	print(f"{c.g}INFO : obs pressure is {hp.OBS_PRESSURE}{c.d}")

	sim = SpecSimulator(hp)
	sim.run()

elif "test" not in sys.argv:

	hp = Hparams(var_params=full_var_train, with_noise=noisy)
	sim = SpecSimulator(hp)
	sim.run()

else:

	hp = Hparams(var_params=full_var_test, with_noise=noisy)
	sim = SpecSimulator(hp)
	sim.run()



