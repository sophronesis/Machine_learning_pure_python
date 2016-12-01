from ml_algo import LinearRegression, linear_func_with_noise 
import numpy as np
from random import seed, uniform 
from matplotlib import pylab as plt

seed(12345)

def main():
	print('Using linear regression on some linear function with noise:')
	#init random func generator
	k,b,start,end,randrange,num = 3,15,-10,10,20,200
	X = np.array([[uniform(start, end)] for i in range(num)])
	y = np.array([linear_func_with_noise(i,k,b,randrange) for i in X])

	#using linear regression
	model = LinearRegression(0.05,1)
	model.fit(X,y,100,1e-5)

	#output results
	print (' Expected: k={:<7} b={:<7}'.format(float(k),float(b)))
	b,k = model.export_model()
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
