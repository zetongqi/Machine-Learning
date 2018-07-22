import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def circle(x, y):
	return (x-5)**2 + (y-5)**2 - 3**2

def metropolis_hastings(D, iter=1000):
	x, y = 3., 3.
	samples = np.zeros((iter, 2))
	for i in range(iter):
		x_next, y_next = np.array([x, y]) + np.random.normal(size=2)
		if np.random.rand() < D(x_next, y_next)/D(x, y):
			x, y = x_next, y_next
		samples[i] = np.array([x, y])
	return samples


samples = metropolis_hastings(circle, iter=100000)
sns.jointplot(samples[:, 0], samples[:, 1])
plt.show()
