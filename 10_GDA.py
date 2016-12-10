import numpy as np
from math import pi
from ml_algo import GDA
from matplotlib import pylab as plt


def main():
	threshold = lambda x: 0 if x<0 else 1
	size = 40
	X = np.r_[([10,5] + np.random.normal(size=(size,2))),([13,4] + np.random.normal(size=(size,2)))]
	y = np.array([0 for i in range(size)]+[1 for i in range(size)])
	model = GDA()
	model.fit(X,y)
	_, plots = plt.subplots(1, 1)
	plots.plot(*np.array(X[:size]).T,'b.')
	plots.plot(*np.array(X[size:]).T,'r.')
	plots.set_title('GDA Classification task')
	#contours
	x = np.linspace(0, 10, 100)
	y = np.linspace(6, 17, 100)
	X, Y = np.meshgrid(y, x)
	Z = np.zeros((len(y),len(x)))
	for i in range(len(x)):
		for j in range(len(y)):
			Z[j][i] = model.predict_prob(np.array([X[j][i],Y[j][i]])) 
	Z0 = np.zeros((len(y),len(x)))
	for i in range(len(x)):
		for j in range(len(y)):
			Z0[j][i] = model.predict_prob_0(np.array([X[j][i],Y[j][i]])) 
	Z1 = np.zeros((len(y),len(x)))
	for i in range(len(x)):
		for j in range(len(y)):
			Z1[j][i] = model.predict_prob_1(np.array([X[j][i],Y[j][i]])) 
	levels = np.linspace(0, 1, 5)
	levels0 = np.linspace(0, 1, 25)
	levels1 = np.linspace(0, 1, 25)
	cs = plots.contour(X, Y, Z, levels=levels)
	cs0 = plots.contour(X, Y, Z0, levels=levels0,colors=['b']*25)
	cs1 = plots.contour(X, Y, Z1, levels=levels1,colors=['r']*25)
	plots.clabel(cs, inline=1, fontsize=10)
	plt.show()
if __name__ == '__main__':
	main()
