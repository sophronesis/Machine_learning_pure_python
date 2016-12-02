import numpy as np
from ml_algo import PCA, sphere_cluster
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
	num = 1000
	first_blob = np.array([sphere_cluster(1,3,7,1) for i in range(num//3)])
	second_blob = np.array([sphere_cluster(4,3,3,2) for i in range(num//3)])
	third_blob = np.array([sphere_cluster(6,1,3,1) for i in range(num//3)])
	X = np.concatenate((first_blob,second_blob,third_blob))
	model = PCA(2)
	model.fit(X)
	new_coords = model.predict_many(X)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(*np.transpose(X))
	ax.set_title('Before PCA')
	for angle in range(0, 360):
	    ax.view_init(30, angle)
	    plt.draw()
	plt.show()
	plt.plot(*np.transpose(new_coords),'o')
	plt.title('After PCA')
	plt.show()



if __name__ == '__main__':
	main()