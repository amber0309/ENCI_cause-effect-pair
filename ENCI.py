"""
ENCI (Embedding-based Nonstationary Causal Model Inference) python implementation
(Anaconda3 5.0.1 64-bit for python 3.6.3 on Windows 10)
Shoubo (shoubo.hu AT gmail.com)
12/12/2017
USAGE:
  direction = cd_enci(XY, al)
 
INPUT:
  XY          - input data, list of numpy arrays. rows of each array are 
               i.i.d. samples, column of each array represent variables
  al          - significance level of HSIC independence test
 
OUTPUT: 
  direction   -  1,  X causes Y
                -1,  Y causes X
 
"""
from __future__ import division
import numpy as np
from random import choice
from HSIC import hsic_gam, rbf_dot
from sklearn import linear_model

class KEMDOPERATION:

	@staticmethod
	def kernel_embedding_K(dist, theta, delta):
		Len = len(dist)
		
		m,n = dist[0].shape 
		
		y = np.ones((m,n), dtype = float)
		
		for i in range(0, Len):
			
			d = dist[i]
			l = delta[0,i]
			a = 0.5
			y5 = np.exp(-d**2/(l)**2)
		
			y = y * y5
			
		y = theta * y
		
		return y 
		
	@staticmethod
	def kernel_embedding_D(data, data_sr, feature_type):
	   
		len1  = len(data)
		len2 = len(data_sr)
		
		xx1 = np.transpose(data)
		xx2 = np.transpose(data_sr)
		
		temp = []
		
		for x in xx1:
			temp.append(x.tolist())
		xx1 = temp 
		
		temp = []
		for x in xx2: 
		   temp.append(x.tolist())
		xx2 = temp 
		
		
		num_of_feature = len(feature_type)
		K = []
		#print num_of_feature        
		for i in range(0, num_of_feature):
			K_k = np.zeros((len1, len2), dtype = float)
			K.append(K_k)
		
		dist_x1_x2 = 0.0 
		
		for i in range(0, len1):
			for j in range(0,len2):
				for k in range(0, num_of_feature):
				
					Type = feature_type[k]
					x1 = xx1[k]
					x2 = xx2[k]
				
					if Type == 'numeric':
						dist_x1_x2 = abs(x1[i] - x2[j])# ** 2 
					elif Type == 'Categorical':
						dist_x1_x2 = float(x1[i]==x2[j])
					else:
						dist_x1_x2 = 0.0 
				
					K[k][i][j] = dist_x1_x2 
		return K 
		
	@staticmethod
	def median_dist(S1, S2, feature_type):
		L1 = len(S1[:,0])
		L2 = len(S2[:,0])
		num_of_feature = len(feature_type)
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			M = []
			for i in range(0, L2):
				for p in range(0, L1):
				
					if feature_type[t] == 'numeric':
						d = np.abs(S1[p,t] - S2[i,t])
					elif feature_type == 'Categorical':
						d = float(S1[p,t] == S2[i,t])
					else: 
						d = 0.0 
				
					M.append(d)
			MM[0,t] = np.median(M)
		return MM
		
	@staticmethod
	def mean_dist(S1, S2, feature_type):
		
		L = len(S1[:,0])
		num_of_feature = len(feature_type)
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			M = []
			for i in range(0, L):
				for p in range(0, i):
				
					if feature_type[t] == 'numeric':
						d = np.abs(S1[p,t] - S2[i,t])
					elif feature_type == 'Categorical':
						d = float(S1[p,t] == S2[i,t])
					else: 
						d = 0.0 
				
					M.append(d)
			MM[0,t] = np.mean(M)
		return MM


def pre_dlt(XY, nsamp = 500):
	"""
	compute the universal kernel width
	INPUT
		XY	- list of numpy arrays ( n * (dx + dy) )
		nsamp	- number of obs used to estimate kernel width
	OUTPUT
		dlt	- list of kernel width of each variable
	"""
	dim = XY[0].shape[1]
	feature_type = ['numeric']

	xyall = np.concatenate( XY, axis=0 )

	xyall = np.random.permutation(xyall)

	dlt = []
	for j in range(0, dim):
		dlt.append(KEMDOPERATION.median_dist(xyall[0:nsamp,j].reshape(-1,1), xyall[0:nsamp,j].reshape(-1,1), feature_type))

	return dlt


def pre_tensor(XY):
	"""
	compute $\tau_x$ and $\tau_y$ of each group
	INPUT
		XY		- list of numpy arrays, each array is a n by (dx + dy) matrix
	OUTPUT
		tau_x, tau_y	- two numpy arrays corresponds to $\tau_x$ and $\tau_y$ of each group
	"""
	N_grp = len(XY)

	feature_type = ['numeric']
	Llist = []

	for k in range(0, N_grp):
		XY[k] = (XY[k] - np.mean(XY[k], axis=0)) / np.std(XY[k], axis=0)

	dlt = pre_dlt(XY, 1000)

	tau_x = np.zeros((N_grp, 1))
	tau_y = np.zeros((N_grp, 1))

	for keridx in range(0, N_grp):
		xy = XY[keridx]
		L, Dim = xy.shape
		x = xy[:,0].reshape(L, 1)
		y = xy[:,1].reshape(L, 1)

		H = np.identity(L) - np.ones((L)) / L

		d_x = KEMDOPERATION.kernel_embedding_D(x, x, feature_type)
		d_y = KEMDOPERATION.kernel_embedding_D(y, y, feature_type)

		k_x_i = KEMDOPERATION.kernel_embedding_K(d_x, 1, dlt[0])
		k_y_i = KEMDOPERATION.kernel_embedding_K(d_y, 1, dlt[1])

		tau_xi = np.trace(np.dot(k_x_i, H))/ L / L
		tau_yi = np.trace(np.dot(k_y_i, H))/ L / L

		tau_x[keridx, 0] = tau_xi
		tau_y[keridx, 0] = tau_yi

		if keridx == 0:
			tau_x_mean = tau_xi
			tau_y_mean = tau_yi
		else:
			tau_x_mean = tau_x_mean + tau_xi
			tau_y_mean = tau_y_mean + tau_yi

		Llist.append(L)

	tau_x_mean = tau_x_mean / N_grp
	tau_y_mean = tau_y_mean / N_grp

	tau_x = tau_x - tau_x_mean
	tau_y = tau_y - tau_y_mean

	return (tau_x, tau_y)


def cd_enci(XY, al = 0.05):
	"""
	infer causal direction using independence test (HSIC)
	
	(Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B., & Smola, A. J. (2008). 
	A kernel statistical test of independence. In Advances in neural information processing systems (pp. 585-592).)
	
	INPUT
		XY	- list of numpy arrays ( n * (dx + dy) )
		al	- significance level of HSIC test
	OUTPUT
		1 (-1)	- X causes Y (Y causes X)
	"""
	[tau_x, tau_y] = pre_tensor(XY)

	regr_x2y = linear_model.LinearRegression()
	regr_y2x = linear_model.LinearRegression()

	# ----- compute residual and conduct HSIC test for x-->y
	regr_x2y.fit(tau_x, tau_y)
	resi_x2y = tau_y - regr_x2y.predict(tau_x)
	[stat_xy, thre_xy] = hsic_gam(tau_x, resi_x2y, al)
	x2y = stat_xy / thre_xy

	# ----- compute residual and conduct HSIC test for y-->x
	regr_y2x.fit(tau_y, tau_x)
	resi_y2x = tau_x - regr_y2x.predict(tau_y)
	[stat_yx, thre_yx] = hsic_gam(tau_y, resi_y2x, al)
	y2x = stat_yx / thre_yx

	if x2y < y2x:
		print('x2y = ' + str(x2y))
		print('y2x = ' + str(y2x))
		print('The causal direction is X --> Y.\n')
		return 1
	else:
		print('x2y = ' + str(x2y))
		print('y2x = ' + str(y2x))
		print('The causal direction is Y --> X.\n')
		return -1

def kurtosis(X):
	s = np.std(X)
	m = np.mean(X)
	k = np.mean((X-m)**4)/(s**4)

	return k

def cd_enci_plingam(XY):
	"""
	infer causal direction using pairwiseLiNGAM
	
	Hyvärinen, A., & Smith, S. M. (2013). Pairwise likelihood ratios 
	for estimation of non-Gaussian structural equation models. Journal 
	of Machine Learning Research, 14(Jan), 111-152.
	
	INPUT
		XY		- list of numpy arrays ( n * (dx + dy) )
	OUTPUT
		1 (-1)	- X causes Y (Y causes X)
	"""

	[tau_x, tau_y] = pre_tensor(XY)

	rho = np.mean(tau_x * tau_y) * np.sign(kurtosis(tau_x))
	R = rho * np.mean(tau_x**3 * tau_y - tau_x * tau_y**3)

	if R > 0:
		print('R = ' + str(R))
		print('The causal direction is X --> Y.\n')
		return 1
	else:
		print('R = ' + str(R))
		print('The causal direction is Y --> X.\n')
		return -1
