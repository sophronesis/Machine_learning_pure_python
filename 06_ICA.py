import numpy as np
from random import uniform, seed
from matplotlib import pyplot as plt
from ml_algo import ICA
#seed(0)

def main():
	n_samples = 8000
	time = np.linspace(0, 8, n_samples)
	s1 = np.sin(time - 1.5)
	s2 = np.sign(np.sin(7 * time + 2.7))
	s3 = np.sin(3 * time + 10)
	s4 = np.arcsin(np.sin(20 * time + 3.85))
	S = np.c_[s1, s2, s3, s4]
	S += 0.05 * np.random.normal(size=S.shape)

	S /= S.std(axis=0)
	A = np.array([[1, 1, 1, 0.7], [0.5, 2, 1.0, 1.3], [1.5, 1.0, 2.0, 0.4], [1.8, 1.7, 0.4, 0.6] ])  # Mixing matrix
	X = np.dot(S, A.T)

	model = ICA(tolerance=1e-5)
	Shat = model.recover_sources(X)
	Shat /= Shat.std(axis=0)
	models = [X, S, Shat]
	names = ['Observations (mixed signal)',
			 'True Sources',
			 'ICA recovered signals']
	colors = ['green', 'red', 'steelblue', 'orange']

	for ii, (model, name) in enumerate(zip(models, names), 1):
		plt.subplot(3, 1, ii)
		plt.title(name)
		for sig, color in zip(model.T, colors):
			plt.plot(sig, color=color)
	plt.show()

if __name__ == '__main__':
	main()


