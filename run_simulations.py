#!/usr/bin/env python
# coding: utf-8

import time
#import IPython
import os
import json
from datetime import datetime
# from pyswarm import pso
# import joblib
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import tensorflow.compat.v2 as tf
# import tensorflow_probability as tfp
# from tensorflow import keras
# import keras.backend as K

from tester.tester import Tester



def run_simulation(x):
	tester = Tester()

	model_input = np.array(x)

	assert len(model_input.shape) == 1

	filepath = os.path.join('.', 'advanced-bayesian-opt', 'input_configuration.csv')

	with open(filepath, 'w') as file:
		file.write('TRIMBG,string,enum,0b1000000\n')
		file.write('TRIMCUR,string,enum,0b10000 \n')
		file.write('models,string,enum,nom \n')
		file.write('vref,string,enum,0.6 \n')
		file.write('vss,string,enum,0v00 \n')
		file.write("vdda_evr,string,enum,{} \n".format(model_input[0]))
		file.write("vdda_hpbg,string,enum,{} \n".format(model_input[0]))
		file.write("vddpd,string,enum,{} \n".format(model_input[1]))
		file.write("T,string,enum,{} \n".format(model_input[2]))

	tester.load_limits(os.path.join('.', 'advanced-bayesian-opt', 'configuration', 'limits.txt'))
	print('\nLoaded limits:\n{}'.format(tester.limits))

	tester.load_in_values(os.path.join('.', 'advanced-bayesian-opt', 'input_configuration.csv'))
	print('\nLoaded in-values:\n{}'.format(tester.in_values))
	print(tester.runs)


	timeout_reached = True
	while timeout_reached:

			tester.run_simulation()
			tester.load_out_results()
			timeout_reached = tester.timeout_reached
			if timeout_reached:
				print("Timeout! Trying again. \n")

	# print('\nSimulation launched!')

	# NOTE: comment out the next two lines if load_in_values() has been invoked with 'create_simlist=False' above
	# tester.run_simulation()
	# print('\nSimulation launched!')

	# tester.load_out_results()

	print('\nLoaded out-results:\n{}'.format(tester.out_results))

	output=tester.out_results['RUN1']['V_ref_1v133_untrimmed']
	return output


points_path = os.path.join('.', 'advanced-bayesian-opt', 'Predictions', 'training_points27.npyz')
vals_path = os.path.join('.', 'advanced-bayesian-opt', 'Predictions','training_values27.npyz')

points=[]
with open(points_path, 'rb') as file:
	data=np.load(file)
	points=data['points']


real_values=[]

for point in points:
	val=run_simulation(point)
	real_values.append(val)


to_save={
'values':real_values,
}

with open(vals_path, 'wb') as file:
	np.savez(file,**to_save)
	


points_path = os.path.join('.', 'advanced-bayesian-opt', 'Predictions', 'training_points64.npyz')
vals_path = os.path.join('.', 'advanced-bayesian-opt', 'Predictions','training_values64.npyz')

points=[]
with open(points_path, 'rb') as file:
	data=np.load(file)
	points=data['points']


real_values=[]

for point in points:
	val=run_simulation(point)
	real_values.append(val)


to_save={
'values':real_values,
}

with open(vals_path, 'wb') as file:
	np.savez(file,**to_save)




