from __future__ import division
import numpy as np
from random import choice
from ENCI import cd_enci, cd_enci_plingam

def expe_dcon(N_grp):
	"""
	an simple example to show the usage of cd_enci()
	N_grp - number of groups of the synthetic data
	"""
	
	XY = []
	for i in range(0, N_grp):
	# generate synthetic data groups and save in list XY
		sample_size = choice(np.arange(40, 50))
		xy = Generate_XYmltp(np.random.randint(7), sample_size)

		XY.append(xy)

	# conduct ENCI with HSIC measure on XY
	order1 = cd_enci(XY)


def Generate_XY(label, sample_size):
	"""
	generate one group of synthetic data in additive mechanism
	INPUT
		label	- type of the causal mechanism ( 0 - 6)
	OUTPUT
		xy	- a data group in a numpy array ( n * (dx + dy) ) 
	"""
	ncoeff = 1
	
	wt = np.random.rand(3) + 0.5
	wt = wt/np.sum(wt)

	L1 = int(wt[0] * sample_size)
	x1 = 0.3 * np.random.randn(L1, 1) - 1
	L2 = int(wt[1] * sample_size)
	x2 = 0.3 * np.random.randn(L2, 1) + 1
	L3 = sample_size - L1 - L2
	x3 = 0.3 * np.random.randn(L3, 1)

	x = np.concatenate((x1, x2, x3), axis = 0)
	c = 0.4 * np.random.rand(1) + 0.8

	if label == 0:
		n = np.random.randn(sample_size, 1)
		y = 1 / (x**2 + 1) + n * ncoeff
	elif label == 1:
		n = np.random.randn(sample_size ,1)
		y = np.sign(c * x) * ((c * x)**2) + n * ncoeff
	elif label == 2:
		n = - np.random.randn(sample_size, 1)
		y = np.cos(c * x * n) + n * ncoeff
	elif label == 3:
		n = np.random.randn(sample_size, 1)
		y = np.sin(c * x) + n * ncoeff
	elif label == 4:
		n = np.random.randn(sample_size, 1)
		y = x**2 + n * ncoeff
	elif label == 5:
		n = np.random.randn(sample_size, 1)
		y = 2*np.sin(x) + 2*np.cos(x) + n * ncoeff
	elif label == 6:
		n = np.random.randn(sample_size, 1)
		y = 4 * np.sqrt(np.abs(x)) + n * ncoeff
	else:
		pass

	xy = np.concatenate((x, y), axis = 1)

	return xy

def Generate_XYmltp(label, sample_size):
	"""
	generate one group of synthetic data in multiplicative mechanism
	INPUT
		label	- type of the causal mechanism ( 0 - 6)
	OUTPUT
		xy	- a data group in a numpy array ( n * (dx + dy) ) 
	"""
	ncoeff = 1
	
	wt = np.random.rand(3) + 0.5
	wt = wt/np.sum(wt)

	L1 = int(wt[0] * sample_size)
	x1 = 0.3 * np.random.randn(L1, 1) - 1
	L2 = int(wt[1] * sample_size)
	x2 = 0.3 * np.random.randn(L2, 1) + 1
	L3 = sample_size - L1 - L2
	x3 = 0.3 * np.random.randn(L3, 1)

	x = np.concatenate((x1, x2, x3), axis = 0)
	c = 0.4 * np.random.rand(1) + 0.8

	if label == 0:
		n = np.random.randn(sample_size, 1)
		y = (1 / (x**2 + 1) ) * n * ncoeff
	elif label == 1:
		n = np.random.randn(sample_size ,1)
		y = np.sign(c * x) * ((c * x)**2) * n * ncoeff
	elif label == 2:
		n = - np.random.randn(sample_size, 1)
		y = np.cos(c * x * n) * n * ncoeff
	elif label == 3:
		n = np.random.randn(sample_size, 1)
		y = np.sin(c * x) * n * ncoeff
	elif label == 4:
		n = np.random.randn(sample_size, 1)
		y = x**2 * n * ncoeff
	elif label == 5:
		n = np.random.randn(sample_size, 1)
		y = (2*np.sin(x) + 2*np.cos(x)) * n * ncoeff
	elif label == 6:
		n = np.random.randn(sample_size, 1)
		y = 4 * np.sqrt(np.abs(x)) * n * ncoeff
	else:
		pass

	xy = np.concatenate((x, y), axis = 1)

	return xy

if __name__ == '__main__':
	expe_dcon(50)
