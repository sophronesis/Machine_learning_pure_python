import numpy as np
from random import uniform, triangular, seed, shuffle
from math import e, log, cos, sin, pi

class LinearRegression():
	def __init__(self,alpha,n,normalize=True,Lambda=0.01,momentum=0.5):
		self.theta = np.array([0 for i in range(n+1)])
		self.alpha = alpha
		self.normalize = normalize
		self.Lambda = Lambda
		self.momentum = momentum
	def fit(self,X,y,iterations,tolerance):
		if self.normalize:
			self.pca = PCA(len(X[0]))
			self.pca.fit(X)
			X = self.pca.predict_many(X)
		leading_one = np.array([1 for i in X])
		X = np.column_stack([leading_one,X])
		self.cost_func_log = []
		self.grad_func_log = []
		prev_grad = np.array([0. for i in X[0]])
		for _ in range(iterations):
			new_theta = np.array(self.theta)
			grad = self.alpha*self.cost_function_deriv(X,y,new_theta) + self.momentum*prev_grad
			new_theta = new_theta - grad
			prev_grad = grad
			self.grad_func_log.append(grad)
			self.theta = new_theta
			self.cost_func_log.append(self.cost_function(X,y,self.theta))
	def _predict(self,x,theta):
		return np.vdot(theta,x)
	def predict(self,x):
		x = (x,)
		x = np.array(x)
		if self.normalize:
			x = self.pca.predict(x)
		if len(x) not in [len(self.theta)-1,len(self.theta)]:
			raise Exception()			
		if len(x)==len(self.theta)-1:
			x = np.concatenate([[1],x])
		return self._predict(x,self.theta)
	def cost_function(self,X,y,theta):
		return (sum([(self._predict(X[i],theta) - y[i])**2 for i in range(len(X))])+self.Lambda*np.vdot(theta,theta))/(2*len(X))
	def cost_function_deriv(self,X,y,theta):
		return (sum([(self._predict(X[i],theta) - y[i])*X[i] for i in range(len(X))]) + self.Lambda*theta)/len(X)
	def export_model(self):
		return self.theta

class BinaryLogisticRegression():
	def __init__(self,alpha,n,normalize=True,Lambda=0.01,momentum=0.5):
		self.theta = np.array([0 for i in range(n+1)])
		self.alpha = alpha
		self.normalize = normalize
		self.Lambda = Lambda
		self.momentum = momentum
	def sigmoid(self,x):
		return 1/(1+e**(-x))
	def fit(self,X,y,iterations,tolerance):
		if self.normalize:
			self.pca = PCA(len(X[0]))
			self.pca.fit(X)
			X = self.pca.predict_many(X)
		leading_one = np.array([1 for i in X])
		X = np.column_stack([leading_one,X])
		self.cost_func_log = []
		self.grad_func_log = []
		prev_grad = np.array([0. for i in X[0]])
		for _ in range(iterations):
			new_theta = np.array(self.theta)
			grad = self.alpha*self.J_func_deriv(X,y,new_theta) + self.momentum*prev_grad
			new_theta = new_theta - grad
			prev_grad = grad
			self.grad_func_log.append(grad)
			self.theta = new_theta
			self.cost_func_log.append(self.J_func(X,y,self.theta))
	def _predict(self,x):
		return self.sigmoid(np.vdot(self.theta,x))
	def predict(self,x):
		x = np.array(x)
		if self.normalize:
			x = self.pca.predict(x)
		if len(x) not in [len(self.theta)-1,len(self.theta)]:
			raise Exception()			
		if len(x)==len(self.theta)-1:
			x = np.concatenate([[1],x])
		return self._predict(x)
	def cost_func(self,x,y):
		value = self._predict(x)
		value = 1e-12 if value==0 else 1-1e-12 if value==1 else value
		return -log(value) if y==1 else -log(1-value)
	def J_func(self,X,y,theta):
		return (sum(self.cost_func(X[i],y[i]) for i in range(len(X))) + self.Lambda*np.vdot(theta,theta))/len(X)
	def J_func_deriv(self,X,y,theta):
		return (sum([(self._predict(X[i]) - y[i])*X[i] for i in range(len(X))]) + self.Lambda*theta)/len(X)
	def export_model(self):
		return self.theta

class KMeansClustering():
	def __init__(self,k):
		self.k = k
	def fit(self,X,initoverexamples=True):
		X = np.array(X)
		self.centroids = np.zeros((self.k,len(X[0])))
		self.cost_func_log = []
		if len(X)<self.k: 
			raise
		if initoverexamples:#init centroids from random input points
			nums = list(range(len(X)))
			shuffle(nums)
			self.centroids = X[nums[:self.k]]
		while True:
			self.c = [[] for i in range(self.k)]
			for num,item in enumerate(X):
				self.c[self.predict(item)].append(num)
			new_centroids = np.array([np.average(X[i],axis=0) for i in self.c])
			self.cost_func_log.append(self.J_func(X))
			if np.sum(new_centroids)==np.sum(self.centroids): break				
			self.centroids=new_centroids
	def predict(self,x):
		scores = {np.linalg.norm(x-self.centroids[i]):i for i in range(len(self.centroids))}
		return scores[min(scores)]
	def J_func(self,X):
		return sum(np.linalg.norm(X[i]-self.centroids[self.predict(X[i])]) for i in range(len(X)))/len(X)
	def export_model(self):
		return self.centroids

class PCA():
	def __init__(self,k=-1,variance_retained=0.95):
		"""if k=-1 pick k automaticaly"""
		self.k = k
		self.variance_retained = variance_retained
	def fit(self,X):
		X = np.array(X)
		#normalization
		mean = np.mean(X,axis=0)
		sigma = np.std(X,axis=0)
		for i in range(len(X[0])):
			for j in range(len(X)):
				X[j][i] = (X[j][i] - mean[i])/sigma[i]
		#computing sigma and U
		sigma = np.dot(np.transpose(X),X)/len(X)
		U,S,V = np.linalg.svd(sigma)
		if self.k==-1:#auto pick k value
			total = np.trace(S)
			summ = 0
			for i in range(len(S)):
				summ+=S[i][i]
				if summ/total >= self.variance_retained:
					self.k = i+1
					break
		self.Ureduced = np.transpose(U[:,:self.k])
	def predict(self,x):
		x = np.transpose(np.array(x))
		return np.dot(self.Ureduced,x)
	def predict_many(self,X):
		X = np.transpose(np.array(X))
		return np.transpose(np.dot(self.Ureduced,X))
	def export_model():
		return self.Ureduced

def circle_cluster(center_x,center_y,rad):
	"""Create sample from shape of a circle in coords (center_x,center_y) and with rad radius"""
	rad,angle = rad*(1-triangular(0,1,0)),uniform(0,2*pi)
	return center_x+rad*cos(angle),center_y+rad*sin(angle)	

def sphere_cluster(center_x,center_y,center_z,rad):
	"""Create sample from shape of a sphere in coords (center_x,center_y,center_z) and with rad radius"""
	while True:#http://mathworld.wolfram.com/SpherePointPicking.html
		x,y,z=rad*uniform(-1,1),rad*uniform(-1,1),rad*uniform(-1,1)
		if x**2+y**2+z**2<rad:
			break
	return center_x+x,center_y+y,center_z+z

def linear_func_with_noise(x,k,b,epsilon):
	"""
	k,b - line parameters
	epsilon - range of noise
	"""
	return k*x+b+uniform(-epsilon,epsilon)