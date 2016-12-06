import numpy as np
from random import uniform
from math import e



class ICA():#https://en.wikipedia.org/wiki/FastICA
	def __init__(self):
		pass
	def fit(self,X):
		#center data
		m,n = len(X),len(X[0])
		X = X - X.mean(axis=0)
		#X = np.transpose(X)
		#whiten data
		g = np.vectorize(lambda x:x*e**(-u**2/2))
		g_p = np.vectorize(lambda x:(1-x**2)*e**(-u**2/2))
		w, v = np.linalg.eig(X)
		D = np.zeros((len(w),len(w)))
		di = np.diag_indices(len(w))
		D[di] = 1. / np.sqrt(w)
		X = v.dot(D).dot(v.T).dot(X)
		W = np.zeros((n,n))
		for i in range(n):
			w = np.random.normal(size=(1,n))
			while True:
				w_old = np.copy(w)
				w = X.dot(g(w.T.dot(X)))/m + g_p()
	            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w # change values
				
	            gwtx, g_wtx = g(fast_dot(w.T, X), fun_args)#get g(mxn) and g_p(1xm) kxp-> kxp+1xk  1xm-> 1xm+1x1 
	            w = np.dot(np.dot(w, W[:j].T), W[:j])#decorelate
	            w1 /= np.sqrt((w1 ** 2).sum())#normalize

	def predict(self):
		pass

def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """Deflationary FastICA using fun approx to neg-entropy function
    Used internally by FastICA.
    """
    n_components = w_init.shape[0]
    w_init = np.random.normal(size=(n_components,
                            n_components))
    #X = nxm - n - source + m - samp 
    W = np.zeros((n_components, n_components), dtype=X.dtype)#nxn
    n_iter = []
    # j is the index of the extracted component
    for j in range(n_components):#from 0 to n
        w = w_init[j, :].copy()#take value from init (w = nx1)
        w /= np.sqrt((w ** 2).sum())#normalize
        for i in moves.xrange(max_iter):#inf cycle
            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break
        n_iter.append(i + 1)
        W[j, :] = w

    return W, max(n_iter)

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

###############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(tol = 1e-6)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

###############################################################################
# Plot results

plt.figure()

models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals', 
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
"""