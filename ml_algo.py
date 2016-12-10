import numpy as np
from random import uniform, triangular, seed, shuffle
from math import e, log, cos, sin, pi
from functools import partial
import re

class FeatureConstructor():
	def __init__(self,features,save_old_vals=True):
		self.features = features
		self.save_old_vals = save_old_vals
	def __call__(self,X):
		X = np.array(X)
		if isinstance(self.features,list) or isinstance(self.features,tuple):
			if len(X.shape)==2:
				new_features = np.array([[feature(item) for feature in self.features] for item in X])
			else:
				new_features = np.array([feature(X) for feature in self.features])
		elif callable(self.features):
			new_features = np.array([self.features(item) for item in X])
		else:
			raise
		if self.save_old_vals:
			if len(X.shape)==len(new_features.shape)==1:
				new_features = np.concatenate([X,new_features])
			else:
				new_features = np.column_stack([X,new_features])
		return new_features

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
		try:
			return self.sigmoid(np.vdot(self.theta,x))
		except Exception:
			print(x,self.theta)
			raise
		else:
			pass
		finally:
			pass
		
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

class Kernel():
	@staticmethod
	def linear():
		return lambda x,y: 1 if x is y else np.dot(x, y)

	@staticmethod
	def gaussian(sigma):
		return lambda x,y: 1 if x is y else np.exp(-np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * sigma ** 2)))

class BinarySVM():
	def __init__(self,kernel=Kernel.linear(),C=10,alpha=0.01,normalize=True,momentum=0.5):
		self.C = C
		self.kernel = kernel
		self.normalize = normalize
		self.momentum = momentum
		self.alpha = alpha
	def fit(self,X,y,iterations):
		self.theta = np.zeros((len(X)+1))
		if self.normalize:
			self.pca = PCA(len(X[0]))
			self.pca.fit(X)
			X = self.pca.predict_many(X)
		self.X = X
		self.cost_func_log = []
		self.grad_func_log = []
		prev_grad = np.array([0. for i in range(len(X)+1)])
		for _ in range(iterations):
			new_theta = np.array(self.theta)
			grad = self.alpha*self.J_func_deriv_numerical(X,y,new_theta) + self.momentum*prev_grad
			new_theta = new_theta - grad
			print('theta=',self.theta)

			print('Grad  num=',self.J_func_deriv_numerical(X,y,new_theta))
			#print('Grad calc=',self.J_func_deriv(X,y,new_theta))
			prev_grad = grad
			self.grad_func_log.append(grad)
			self.theta = new_theta
			self.cost_func_log.append(self.J_func(X,y,self.theta))
	def kernel_features(self,x):
		return np.array([1]+[self.kernel(x,self.X[i]) for i in range(len(self.X))])
	def _predict(self,x,theta):
		return np.vdot(theta,self.kernel_features(x))
	def predict(self,x):
		x = (x,)
		x = np.array(x)
		if self.normalize:
			x = self.pca.predict(x).T
		return self._predict(x,self.theta)
	def predict_many(self,X):
		if self.normalize:
			X = self.pca.predict_many(X).T
		return np.array([self._predict(x,self.theta) for x in X])
	def cost_func(self,x,theta,val_sign):
		y = self._predict(x,theta)
		if val_sign==1:
			return 0 if y>1 else 1-y
		else:
			return 0 if y<-1 else y+1
	def cost_func_deriv(self,x,theta,val_sign):
		y = self._predict(x,theta)
		x = self.kernel_features(x)
		if val_sign==1:
			return np.array([0 if y>1 else -x[i] for i in range(len(x))]) 
		else:
			return np.array([0 if y<-1 else x[i] for i in range(len(x))])
	def J_func(self,X,y,theta):
		return self.C*sum(self.cost_func(X[i],theta,y[i]) for i in range(len(X))) + np.vdot(theta,theta)
	def J_func_deriv_numerical(self,X,y,theta,epsilon=1e-10):
		new_theta = np.copy(theta)
		current_cost = self.J_func(X,y,theta)
		J_func_der = np.zeros(theta.shape)
		for i in range(len(new_theta)):
			new_theta[i] = theta[i] + epsilon
			cost_plus = self.J_func(X,y,new_theta)
			new_theta[i] = theta[i] - epsilon
			cost_minus = self.J_func(X,y,new_theta)
			J_func_der[i] = (cost_plus - cost_minus)/(2*epsilon)
		return J_func_der
	def J_func_deriv(self,X,y,theta):
		print()
		return sum([(self._predict(X[i],theta) - y[i])*self.cost_func_deriv(X[i],theta,y[i]) for i in range(len(X))]) + theta
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

class NaiveBayesClassifier():
	def __init__(self,min_prob=1e-5,punctuation=["#$.,:?!"]):
		self.classnamecount = {}
		self.data = {}
		self.punctuation = punctuation
		self.min_prob = 1e-5
	def _addClass(self,classname):
		if not classname in self.classnamecount:
			self.classnamecount[classname] = 0
		self.classnamecount[classname]+=1
	def _addToken(self,token,classname):
		if not token in self.data:
			self.data[token] = {}
		if not classname in self.data[token]:
			self.data[token][classname] = 0
		self.data[token][classname] +=1
	def _getTotalCount(self):
		return sum(self.classnamecount[i] for i in self.classnamecount)
	def _tokenize(self,string):
		string = re.sub(str(self.punctuation),"",string)	
		tokens = string.lower().split()
		return tokens
	def _getTokenProb(self,token,classname):
		if not token in self.data:
			return None
		if classname in self.classnamecount and classname in self.data[token]:#return probability if presented
			return self.data[token][classname]/self.classnamecount[classname] 
		return self.min_prob
	def train(self,string,classname):
		self._addClass(classname)
		tokens = self._tokenize(string)
		for token in tokens:
			self._addToken(token,classname)
	def predict_prob(self,string):
		from functools import reduce
		tokens = list(set(self._tokenize(string)))
		probPerClass = {}
		for classname in self.classnamecount:
			probPerToken = [self._getTokenProb(token, classname) for token in tokens]
			probPerClass[classname] = reduce(lambda a,b: a*b, [i for i in probPerToken if i]) 
		if len(probPerClass)>0:
			summ = sum(probPerClass.values())
			for item in probPerClass:
				probPerClass[item] /= summ
			return probPerClass
		else: 
			return "Not enough data" 
		return probPerClass 
	def predict_class(self,string):
		probPerClass = self.predict_prob(string)
		maxx = None if not probPerClass else max(probPerClass.keys(), key=lambda key:probPerClass[key])
		return [maxx,probPerClass[maxx]] if maxx else ["Couldn't classify",0]

class ICA():
	def __init__(self,tolerance=1e-5,max_iteration=20):
		self.tolerance = tolerance
		self.max_iteration= max_iteration
	def recover_sources(self,X,w_init=None):
		np.copy(X)
		m,n = len(X),len(X[0])
		#center data
		X = (X - X.mean(axis=0)).T
		#whiten data 
		u, d, _ = np.linalg.svd(X, full_matrices=False)
		K = (u / d).T
		X = np.dot(K, X)
		X *= np.sqrt(X.shape[1])

		W = np.zeros((n,n))
		for j in range(n):
			w = np.random.normal(size=(4,))
			w /= np.sqrt((w ** 2).sum())
			for i in range(self.max_iteration):
				x = np.dot(w.T, X)
				f  = np.exp(-(x ** 2) / 2)
				g  = x * f
				dg = ((1 - x ** 2) * f).mean(axis=-1)
				w1 = (X * g).mean(axis=1) - dg.mean() * w
				w1 -= np.dot(np.dot(w1, W[:j].T), W[:j])
				w1 /= np.sqrt((w1 ** 2).sum())
				lim = np.abs(np.abs((w1 * w).sum()) - 1)
				w = w1
				if lim < self.tolerance:
					break
			W[j, :] = w
		S = np.dot(W, X).T
		return S

class GDA(object):
	def __init__(self):
		pass	
	def fit(self,X,y):
		X0 = np.array([X[n] for n,i in enumerate(y) if i==0])
		X1 = np.array([X[n] for n,i in enumerate(y) if i==1])
		self.X0_mean,self.X1_mean = X0.mean(axis=0), X1.mean(axis=0)
		getSigma = lambda X,mX:(lambda nX:nX.T.dot(nX))(X-mX)/X.shape[0]
		self.X0_sigma,self.X1_sigma = getSigma(X0,self.X0_mean),getSigma(X1,self.X1_mean)
	def predict_prob_0(self,x):
		X0_prob = np.exp(-1/2*(x-self.X0_mean).T.dot(np.linalg.pinv(self.X0_sigma).dot(x-self.X0_mean)))/((2*pi)**(len(x)/2)*np.linalg.det(self.X0_sigma))
		return X0_prob
	def predict_prob_1(self,x):	
		X1_prob = np.exp(-1/2*(x-self.X1_mean).T.dot(np.linalg.pinv(self.X1_sigma).dot(x-self.X1_mean)))/((2*pi)**(len(x)/2)*np.linalg.det(self.X1_sigma))	
		return X1_prob
	def predict_prob(self,x):
		X0_prob,X1_prob = self.predict_prob_0(x),self.predict_prob_1(x)
		return 1-X0_prob/(X0_prob+X1_prob) if X0_prob>X1_prob else X1_prob/(X0_prob+X1_prob)
	def predict(self,x):
		prob = self.predict_prob(x)
		return 0 if prob<0.5 else 1


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