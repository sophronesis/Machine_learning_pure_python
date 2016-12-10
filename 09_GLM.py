import numpy as np
from ml_algo import BinaryLogisticRegression, FeatureConstructor, circle_cluster
from matplotlib import pylab as plt
from random import seed
from math import log
seed(12345)

def main():
	X = [[1,20],[3,40],[5,60],[7,80],[9,100]]
	f = FeatureConstructor([lambda x:x[0]*x[1],lambda x:x[0]**2,lambda x:x[1]**2,lambda x:x[0]-x[1]])
	print("Example of additional feature constructing:\n",f(X))
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
	model1 = BinaryLogisticRegression(0.03,2)
	model1.fit(Xtrain,ytrain,1000,1e-5)
	_Xtrain = f(Xtrain)
	model2 = BinaryLogisticRegression(0.001,len(_Xtrain[0]),momentum=0.5)
	model2.fit(_Xtrain,ytrain,1000,1e-5)

	#preparing visualisation
	threshold = lambda x: 0 if x<0.5 else 1
	confmat1 = np.zeros((2,2))
	confmat2 = np.zeros((2,2))
	confusion_dots1 = [[[],[]],[[],[]]]
	confusion_dots2 = [[[],[]],[[],[]]]
	for i in range(len(Xtest)):
		result = threshold(model1.predict(Xtest[i]))
		confmat1[result][ytest[i]]+=1
		confusion_dots1[result][ytest[i]].append(Xtest[i])
		result = threshold(model2.predict(f(Xtest[i])))
		confmat2[result][ytest[i]]+=1
		confusion_dots2[result][ytest[i]].append(Xtest[i])
	print('Basic features\nConfusion matrix:')
	print(confmat1)
	accuracy = (confmat1[0][0]+confmat1[1][1])/np.sum(confmat1)
	precision = confmat1[0][0]/(confmat1[0][0]+confmat1[1][0])
	recall = confmat1[0][0]/(confmat1[0][0]+confmat1[0][1])
	f1score = 2*precision*recall/(precision+recall)
	print('\nAccuracy:\t{}\nPrecision:\t{}\nRecall:  \t{}\nF1 score:\t{}'.format(accuracy,precision,recall,f1score))
	print('\nSome additional features\nConfusion matrix:')
	print(confmat2)
	accuracy = (confmat2[0][0]+confmat2[1][1])/np.sum(confmat2)
	precision = confmat2[0][0]/(confmat2[0][0]+confmat2[1][0])
	recall = confmat2[0][0]/(confmat2[0][0]+confmat2[0][1])
	f1score = 2*precision*recall/(precision+recall)
	print('\nAccuracy:\t{}\nPrecision:\t{}\nRecall:  \t{}\nF1 score:\t{}'.format(accuracy,precision,recall,f1score))
	print('As we see with more features our algorithm generalizes slightly better')

	#plotting graph
	_, plots = plt.subplots(2, 2)
	plots[0][0].plot(*np.array(confusion_dots1[0][0]).T,'b.')
	plots[0][0].plot(*np.array(confusion_dots1[1][0]).T,'bx')
	plots[0][0].plot(*np.array(confusion_dots1[0][1]).T,'rx')
	plots[0][0].plot(*np.array(confusion_dots1[1][1]).T,'r.')
	plots[0][0].set_title('Classification task (without new features)')
	plots[1][0].plot(*np.array(confusion_dots2[0][0]).T,'b.')
	plots[1][0].plot(*np.array(confusion_dots2[1][0]).T,'bx')
	plots[1][0].plot(*np.array(confusion_dots2[0][1]).T,'rx')
	plots[1][0].plot(*np.array(confusion_dots2[1][1]).T,'r.')
	plots[1][0].set_title('Classification task (with new features)')
	plots[0][1].plot(model1.cost_func_log)
	plots[0][1].set_title('Cost function')
	plots[1][1].plot(model2.cost_func_log)
	plots[1][1].set_title('Cost function')

	#contours
	x = np.linspace(-10, 30, 100)
	y = np.linspace(-10, 30, 100)
	X, Y = np.meshgrid(x, x)
	Z1 = np.zeros((len(y),len(x)))
	for i in range(len(x)):
		for j in range(len(y)):
			Z1[j][i] = model1.predict(np.array([X[j][i],Y[j][i]])) 
	Z2 = np.zeros((len(y),len(x)))
	for i in range(len(x)):
		for j in range(len(y)):
			a = np.array([X[j][i],Y[j][i]])
			#print(a)
			#print(f(a))
			Z2[j][i] = model2.predict(f(a)) 
	levels = np.linspace(0, 1, 3)
	cs = plots[0][0].contour(X, Y, Z1, levels=levels)
	plots[0][0].clabel(cs, inline=1, fontsize=10)
	cs = plots[1][0].contour(X, Y, Z2, levels=levels)
	plots[1][0].clabel(cs, inline=1, fontsize=10)

	plt.show()

if __name__ == '__main__':
	main()

