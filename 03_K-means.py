import numpy as np
from matplotlib import pylab as plt
from numpy import genfromtxt
from random import uniform, seed, shuffle, triangular
from math import e, log, cos, sin, pi
seed(12345)

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
	def export_centroids(self):
		return self.centroids

def circle_cluster(center_x,center_y,rad):
	rad,angle = rad*(1-triangular(0,1,0)),uniform(0,2*pi)
	return center_x+rad*cos(angle),center_y+rad*sin(angle)	
def circluar_clusters():
	num_train = 1000
	red_dots = np.array([circle_cluster(10,10,15) for i in range(num_train//2)])
	blue_dots = np.array([circle_cluster(20,0,15) for i in range(num_train//2)])
	X = np.concatenate((red_dots,blue_dots))
	model = KMeansClustering(2)
	model.fit(X)
	predicted_red = []
	predicted_blue = []
	for i in X:
		if model.predict(i)==0:
			predicted_red.append(i)
		else:
			predicted_blue.append(i)
	print('Centroid coords:')
	centroids = model.export_centroids()
	print(centroids)

	f, plots = plt.subplots(2, 1)
	plots[0].plot(*np.transpose(predicted_red),'r.')
	plots[0].plot(*np.transpose(predicted_blue),'b.')
	plots[0].plot(*np.transpose(centroids),'go')
	plots[0].set_title('Clusterisation task')
	plots[1].plot(model.cost_func_log)
	plots[1].set_title('Cost function')
	plt.show()


def main():
	circluar_clusters()
	#X = np.genfromtxt('iris.csv', delimiter=',')
	#X = X[:,:4]
	#print(X)

if __name__ == '__main__':
	main()