import numpy as np
import random
from arff_parser import *
from sklearn import preprocessing


# return X and Y as numpy array
def feature_scaling(raw):
	X = []
	Y = []
	for row in raw:
		X.append(row[:-1])
		Y.append(row[-1])
	X = np.asarray(X)
	X = preprocessing.scale(X)
	Y = np.asarray(Y)
	return X, Y


def change_label_encoding(Y, label_list):
	for i in range(Y.shape[0]):
		for j in range(len(label_list)):
			if Y[i] == label_list[j]:
				Y[i] = j
	return Y


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def initial_theta(num_of_theta):
	return np.random.uniform(-0.01, 0.01, num_of_theta)


def generate_batch(X, Y, batchsize):
	for i in np.arange(0, len(X), batchsize):
		yield X[i:i+batchsize], Y[i:i+batchsize]


def cross_entropy(h, y):
	# return the average
	return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def gradients(X_batch, Y_batch, theta):
	 z = np.dot(X_batch, theta)
	 h = sigmoid(z)
	 return np.dot(X_batch.T, (h - Y_batch)) / Y_batch.size, h


def process_data(raw, label_list):
	X, Y = feature_scaling(raw)
	Y = change_label_encoding(Y, label_list)
	Y = Y.astype(int)
	return X, Y


def stochastic_gradient_descent(X, Y, e, l, batchsize, X_test, Y_test):
	theta = initial_theta(X.shape[1])
	for epoch in range(e):
		for X_batch, Y_batch in generate_batch(X, Y, batchsize):
			grads, h = gradients(X_batch, Y_batch, theta)
			theta -= l * grads
		error = cross_entropy(h, Y_batch)
		print("epoch: ", epoch, sep="")
		print("error: ", error, sep="")
		print("accuracy: ", get_text_accuracy(X_test, Y_test, theta)*100, "%", sep="")


def get_text_accuracy(X_test, Y_test, theta):
	cnt = 0
	for i in range(X_test.shape[0]):
		h = np.dot(X_test[i], theta.T)
		if h >= 0.5:
			h = 1
		else:
			h = 0
		if h == Y_test[i]:
			cnt += 1
	return cnt / X_test.shape[0]


def fit(raw, label_list, e, l, batchsize, raw_test):
	X, Y = process_data(raw, label_list)
	X_test, Y_test = process_data(raw_test, label_list)
	stochastic_gradient_descent(X, Y, e, l, batchsize, X_test, Y_test)


lst = ["g", "h"]
fit(arff_data("magic_train.arff").data, lst, 10, 0.1, 100, arff_data("magic_test.arff").data)
















