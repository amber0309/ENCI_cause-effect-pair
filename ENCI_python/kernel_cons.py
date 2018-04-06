from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix

class KEMDOPERATION:

	@staticmethod
	def kernel_embedding_K(dist, theta, delta):
		Len = len(dist)
		
		m,n = dist[0].shape 
		
		y = np.ones((m,n), dtype = float)
		
		for i in range(0, Len):
			
			d = dist[i]
			l = delta[0,i]

			# RBF kernel
			y5 = np.exp(-d**2/(2*(l)**2))
		
			y = y * y5
			
		y = theta * y
		
		return y 
		
	@staticmethod
	def kernel_embedding_D(data, data_sr, feature_type):
		"""
		S1, S2         - numpy array
		               - row samples
		               - col dimensions

		output a list of numpy array
		each array is the distance matrix of a dimension
		"""
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
	def kernel_embedding_D_scipy(data, data_sr):
		"""
		S1, S2         - numpy array
		               - row samples
		               - col dimensions

		output a list of numpy array
		each array is the distance matrix of a dimension
		"""
		num_of_feature = data.shape[1]
		D = []
		#print num_of_feature        
		for t in range(0, num_of_feature):
			x_i = data[:,t].reshape(-1,1)
			y_i = data_sr[:,t].reshape(-1,1)
			d_i = distance_matrix(x_i, y_i, p=1, threshold=1000000)
			D.append(d_i)

		return D

	@staticmethod
	def median_dist(S1, S2, feature_type):
		"""
		S1, S2         - numpy array
		               - row samples
		               - col dimensions

		feature_type   - a list

		output a 1 by dim numpy array
		dim is the length of feature_type
		"""
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
	def median_dist_np(S1, S2):
		"""
		S1, S2         - numpy array
		               - row samples
		               - col dimensions

		output a 1 by dim numpy array

		!! works only for numerical arrays
		"""
		num_of_feature = S1.shape[1]
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			b = np.array(np.meshgrid(S1[:,t], S2[:,t])).T.reshape(-1,2)
			abs_diff = abs(b[:,0] - b[:,1])
			#c = abs_diff[abs_diff != 0]
			MM[0,t] = np.median(abs_diff)
		return MM

	@staticmethod
	def median_dist_scipy(S1, S2):
		"""
		S1, S2         - numpy array
		               - row samples
		               - col dimensions

		output a 1 by dim numpy array

		!! works only for numerical arrays
		"""
		num_of_feature = S1.shape[1]
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			x_i = S1[:,t].reshape(-1,1)
			y_i = S2[:,t].reshape(-1,1)
			d_i = distance_matrix(x_i, y_i, p=1, threshold=1000000)
			#c = abs_diff[abs_diff != 0]
			MM[0,t] = np.median(d_i)
		return MM


	@staticmethod
	def mean_dist(S1, S2, feature_type):
		"""
		S1, S2         - numpy array
		               - row samples
		               - col dimensions

		output a 1 by dim numpy array
		"""
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

	@staticmethod
	def mean_dist_np(S1, S2):
		"""
		S1, S2         - numpy array
		               - row samples
		               - col dimensions

		output a 1 by dim numpy array

		!! works only for numerical arrays
		"""
		num_of_feature = S1.shape[1]
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			b = np.array(np.meshgrid(S1[:,t], S2[:,t])).T.reshape(-1,2)
			abs_diff = abs(b[:,0] - b[:,1])
			#c = abs_diff[abs_diff != 0]
			MM[0,t] = np.mean(abs_diff)
		return MM

	@staticmethod
	def mean_dist_scipy(S1, S2):
		"""
		S1, S2         - numpy array
		               - row samples
		               - col dimensions

		output a 1 by dim numpy array

		!! works only for numerical arrays
		"""
		num_of_feature = S1.shape[1]
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			x_i = S1[:,t].reshape(-1,1)
			y_i = S2[:,t].reshape(-1,1)
			d_i = distance_matrix(x_i, y_i, p=1, threshold=1000000)
			#c = abs_diff[abs_diff != 0]
			MM[0,t] = np.mean(d_i)
		return MM
