import numpy as np
import pandas as pd
import math
import pmdarima as pm
import datetime
import os
import warnings

from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss, acf
from collections import defaultdict


def adfuller_test(data, col, trace=False):
	"""
	Perform the Augmented Dickey-Fuller Tests on the Target Data.

	Parameters:
	----------
	data : pandas.DataFrame
		Represents the dataset.

	col : str
		Represents the target data column

	trace : bool, optional; default:False
		Represents whether to showcase results from tests.


	Returns:
	-------
	results : pandas.Series
		Represents the results of the test.

	is_stn : bool
		Represents whether the data is stationary or not.
	"""

	test = adfuller(data[col])
	results = pd.Series(test[0:4], index=['ADF-Test Stat', 'p-value', '# Lags Used', '# Observations'])
	
	for k, v in test[4].items():
		results[f'critical value {k}'] = v

	if trace:
		print(results.to_string(), '\n')

		if test[1] <= 0.05:
			print('The data is stationary by nature.')

		else:
			print('The data is not stationary by nature.')

	is_stn = test[1] <= 0.05

	return results, is_stn


def kpss_test(data, col, reg='c', trace=False):

	"""
	Function to perform the KPSS test on the data

	Parameters:
	----------
	data : pandas.DataFrame
		Represents the dataset

	col : str
		Represents the target column name

	reg : str, optional; default:'c'
		Represents whether the data has any trend or not. Possible values
		`c` and `ct`

	trace : bool, optional; default:False
		Represents whether to showcase the results of the test

	Returns:
	--------

	results: pandas.Series
		Represents the results of the test

	is_stn : bool
		Represents whether the series is stationary or not.

	"""

	test = kpss(data[col], reg)
	results = pd.Series(test[0:3], index=['KPSS-test Stat', 'p-value', '# Lags Used'])

	for k, v in test[3].items():
		results[f'critical value {k}'] = v

	if trace:
		print(results.to_string(), '\n')

		if test[1] > 0.05:
			print('The series is Stationary')

		else:
			print('The Series is Non-Stationary')


	is_stn = test[1] > 0.05

	return results, is_stn