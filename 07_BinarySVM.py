from ml_algo import BinarySVM, circle_cluster
import numpy as np
from matplotlib import pylab as plt
from random import seed
seed(12345)

def main():

	#init random cluster of dots
	num_train = 20
	num_test = 20
	red_dots = np.array([circle_cluster(20,0,5) for i in range(num_train//2)])
	blue_dots = np.array([circle_cluster(10,10,5) for i in range(num_train//2)])
	Xtrain = np.concatenate((red_dots,blue_dots))
	ytrain = np.array([0 for _ in range(num_train//2)]+[1 for _ in range(num_train//2)])
	
	red_dots_test = np.array([circle_cluster(20,0,5) for i in range(num_test//2)])
	blue_dots_test = np.array([circle_cluster(10,10,5) for i in range(num_test//2)])
	Xtest = np.concatenate((red_dots_test,blue_dots_test))
	ytest = np.array([0 for _ in range(num_test//2)]+[1 for _ in range(num_test//2)])
	
	#use logistic regression, calculating on test data and output metrics
	model = BinarySVM()
	model.fit(Xtrain,ytrain,20)
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
	print(model.export_model())
	#plotting graph
	f, plots = plt.subplots(2, 1)
	plots[0].plot(*np.array(confusion_dots[0][0]).T,'r.')
	#plots[0].plot(*np.array(confusion_dots[1][0]).T,'rx')
	#plots[0].plot(*np.array(confusion_dots[0][1]).T,'bx')
	plots[0].plot(*np.array(confusion_dots[1][1]).T,'b.')

	x = np.linspace(0, 30, 100)
	y = np.linspace(-10, 20, 100)
	X, Y = np.meshgrid(x, x)
	print("XY:",X,Y)
	Z = np.zeros((len(y),len(x)))
	for i in range(len(x)):
		for j in range(len(y)):
			Z[j][i] = model.predict(np.array([X[j][i],Y[j][i]])) 

	levels = np.linspace(-1, 1, 20)

	cs = plots[0].contour(X, Y, Z, levels=levels)
	plots[0].clabel(cs, inline=1, fontsize=10)

	plots[0].set_title('Classification task')
	plots[1].plot(model.cost_func_log)
	plots[1].set_title('Cost function (blue curve - with momentum grad)')
	plt.show()

if __name__ == '__main__':
	main()