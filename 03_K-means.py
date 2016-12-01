from ml_algo import KMeansClustering, circle_cluster
import numpy as np
from matplotlib import pylab as plt
from random import seed
seed(12345)

def main():
	num = 1000
	red_dots = np.array([circle_cluster(10,10,15) for i in range(num//2)])
	blue_dots = np.array([circle_cluster(20,0,15) for i in range(num//2)])
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
	centroids = model.export_model()
	print(centroids)

	f, plots = plt.subplots(2, 1)
	plots[0].plot(*np.transpose(predicted_red),'r.')
	plots[0].plot(*np.transpose(predicted_blue),'b.')
	plots[0].plot(*np.transpose(centroids),'go')
	plots[0].set_title('Clusterisation task with kmeans (2 clusters)')
	plots[1].plot(model.cost_func_log)
	plots[1].set_title('Cost function')
	plt.show()


if __name__ == '__main__':
	main()