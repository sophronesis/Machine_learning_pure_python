import numpy as np
from matplotlib import pylab as plt
from random import uniform

class LinearRegression():
	def __init__(self,alpha,n):
		self.theta = np.array([0 for i in range(n+1)])
		self.alpha = alpha
	def fit(self,X,y,iterations,tolerance):
		leading_one = np.array([1 for i in X])
		X = np.column_stack([leading_one,X])
		self.cost_func_log = []
		self.grad_func_log = []
		for _ in range(iterations):
			new_theta = np.array(self.theta)
			grad = sum((i-j)**2 for i,j in zip(self.theta,new_theta))
			new_theta = self.theta - self.alpha*self.cost_function_deriv(X,y)
			self.grad_func_log.append(grad)
			self.theta = new_theta
			self.cost_func_log.append(self.cost_function(X,y))
	def _predict(self,x):
		return np.vdot(self.theta,x)
	def predict(self,x):
		x = np.array([x])
		if x[0]!=1:
			x = np.column_stack([[1],x])
		return self._predict(x)
	def cost_function(self,X,y):
		return sum([(self._predict(X[i]) - y[i])**2 for i in range(len(X))])/(2*len(X))
	def cost_function_deriv(self,X,y):
		return sum([(self._predict(X[i]) - y[i])*X[i] for i in range(len(X))])/len(X)
	def export_params(self):
		return self.theta

def main():
	print('Using linear regression on some linear function with noise:')
	#init random func generator
	k,b,start,end,randmax,randmin,num = 3,15,-10,10,-20,20,200
	func = lambda x,k,b,i,j:k*x+b+uniform(i,j)
	X = np.array([[uniform(start, end)] for i in range(num)])
	y = np.array([func(i,k,b,randmin,randmax) for i in X])

	#using linear regression
	model = LinearRegression(0.05,1)
	model.fit(X,y,100,1e-5)

	#output results
	print (' Expected: k={:<7} b={:<7}'.format(float(k),float(b)))
	b,k = model.export_params()
	print ('Predicted: k={:<7.3} b={:<7.3}'.format(k,b))

	#plotting cost function and data fitting
	f, plots = plt.subplots(2, 1)
	plots[0].plot(X,y,'ro')
	plots[0].plot([start,end],[model.predict(start),model.predict(end)])
	plots[0].set_title('Random data fitting')

	plots[1].plot(model.cost_func_log)
	plots[1].set_title('Cost function')
	plt.show()
if __name__ == '__main__':
	main()

