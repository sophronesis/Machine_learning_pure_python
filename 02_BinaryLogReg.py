import numpy as np
from matplotlib import pylab as plt
from random import uniform, triangular, seed
from math import e, log, cos, sin, pi
seed(12345)

class BinaryLogisticRegression():
	def __init__(self,alpha,n):
		self.theta = np.array([0 for i in range(n+1)])
		self.alpha = alpha
	def sigmoid(self,x):
		return 1/(1+e**(-x))
	def fit(self,X,y,iterations,tolerance):
		leading_one = np.array([1 for i in X])
		X = np.column_stack([leading_one,X])
		self.cost_func_log = []
		self.grad_func_log = []
		for _ in range(iterations):
			new_theta = np.array(self.theta)
			grad = sum((i-j)**2 for i,j in zip(self.theta,new_theta))
			new_theta = self.theta - self.alpha*self.J_func_deriv(X,y)
			self.grad_func_log.append(grad)
			self.theta = new_theta
			self.cost_func_log.append(self.J_func(X,y))
	def _predict(self,x):
		return self.sigmoid(np.vdot(self.theta,x))
	def predict(self,x):
		x = np.array(x)
		if len(x)==len(self.theta)-1:
			x = np.concatenate([[1],x])
			return self._predict(x)
		elif len(x)==len(self.theta):
			return self._predict(x)
		else: 
			raise Exception(x)
	def cost_func(self,x,y):
		value = self._predict(x)
		value = 1e-12 if value==0 else 1-1e-12 if value==1 else value
		return -log(value) if y==1 else -log(1-value)
	def J_func(self,X,y):
		return sum(self.cost_func(X[i],y[i]) for i in range(len(X)))/len(X)
	def J_func_deriv(self,X,y):
		return sum([(self._predict(X[i]) - y[i])*X[i] for i in range(len(X))])
	def export_params(self):
		return self.theta

def circle_cluster(center_x,center_y,rad):
	rad,angle = rad*(1-triangular(0,1,0)),uniform(0,2*pi)
	return center_x+rad*cos(angle),center_y+rad*sin(angle)

def main():

	#init random cluster of dots
	num_train = 400
	num_test = 200
	red_dots = np.array([circle_cluster(10,10,10) for i in range(num_train//2)])
	blue_dots = np.array([circle_cluster(20,0,10) for i in range(num_train//2)])
	Xtrain = np.concatenate((red_dots,blue_dots))
	ytrain = np.array([0 for _ in range(num_train//2)]+[1 for _ in range(num_train//2)])
	
	red_dots_test = np.array([circle_cluster(10,10,10) for i in range(num_test//2)])
	blue_dots_test = np.array([circle_cluster(20,0,10) for i in range(num_test//2)])
	Xtest = np.concatenate((red_dots_test,blue_dots_test))
	ytest = np.array([0 for _ in range(num_test//2)]+[1 for _ in range(num_test//2)])
	
	#use logistic regression, calculating on test data and output metrics
	model = BinaryLogisticRegression(0.0001,2)
	model.fit(Xtrain,ytrain,100,1e-5)
	threshold = lambda x: 0 if x<0.5 else 1
	confmat = np.zeros((2,2))
	confusion_dots = [[[],[]],[[],[]]]
	for i in range(len(Xtest)):
		result = threshold(model.predict(Xtest[i]))
		confmat[result][ytest[i]]+=1
		confusion_dots[result][ytest[i]].append(Xtest[i])
	print('Confusion matrix:')
	print(confmat)
	accuracy = (confmat[0][0]+confmat[1][1])/np.sum(confmat)
	precision = confmat[0][0]/(confmat[0][0]+confmat[1][0])
	recall = confmat[0][0]/(confmat[0][0]+confmat[0][1])
	f1score = 2*precision*recall/(precision+recall)
	print('\nAccuracy:\t{}\nPrecision:\t{}\nRecall:  \t{}\nF1 score:\t{}'.format(accuracy,precision,recall,f1score))

	#plotting graph
	f, plots = plt.subplots(2, 1)
	plots[0].plot(*np.transpose(confusion_dots[0][0]),'r.')
	plots[0].plot(*np.transpose(confusion_dots[1][0]),'rx')
	plots[0].plot(*np.transpose(confusion_dots[0][1]),'bx')
	plots[0].plot(*np.transpose(confusion_dots[1][1]),'b.')
	plots[0].set_title('Classification task')
	plots[1].plot(model.cost_func_log)
	plots[1].set_title('Cost function')
	plt.show()

if __name__ == '__main__':
	main()