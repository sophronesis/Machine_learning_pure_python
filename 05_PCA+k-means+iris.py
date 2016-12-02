from ml_algo import KMeansClustering, PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
import numpy as np
def main():
	#loading data, droping labels and using kmeans
	X = np.genfromtxt('iris.csv', delimiter=',')
	X = X[:,:4]
	kmeans = KMeansClustering(3)
	kmeans.fit(X)
	predicted_red = []
	predicted_blue = []
	predicted_green = []
	predicted = [predicted_red,predicted_green,predicted_blue]
	for i in X:
		predicted[kmeans.predict(i)].append(i)
	#using pca for visualisation purpose
	PCA3d = PCA(3)
	PCA3d.fit(X)
	Z_red_3d = PCA3d.predict_many(predicted_red)
	Z_green_3d = PCA3d.predict_many(predicted_green)
	Z_blue_3d = PCA3d.predict_many(predicted_blue)
	Z_centroids_3d = PCA3d.predict_many(kmeans.export_model())
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(*np.transpose(Z_red_3d),c='r')
	ax.scatter(*np.transpose(Z_green_3d),c='g')
	ax.scatter(*np.transpose(Z_blue_3d),c='b')
	ax.scatter(*np.transpose(Z_centroids_3d),s=80,c='k',marker='x',alpha=1)
	ax.set_title('Using k-means over raw iris data')
	for angle in range(0, 360):
	    ax.view_init(30, angle)
	    plt.draw()
	plt.show()
	PCA2d = PCA(2)
	PCA2d.fit(X)
	Z_red_2d = PCA2d.predict_many(predicted_red)
	Z_green_2d = PCA2d.predict_many(predicted_green)
	Z_blue_2d = PCA2d.predict_many(predicted_blue)
	Z_centroids_2d = PCA2d.predict_many(kmeans.export_model())
	plt.plot(*np.transpose(Z_red_2d),'ro')
	plt.plot(*np.transpose(Z_green_2d),'go')
	plt.plot(*np.transpose(Z_blue_2d),'bo')
	plt.plot(*np.transpose(Z_centroids_2d),'kx',markersize=10)
	plt.title('Using k-means over raw iris data')
	plt.show()



if __name__ == '__main__':
	main()